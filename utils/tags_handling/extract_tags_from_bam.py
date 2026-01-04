import CellTag
import pysam
import argparse
import re
import os
import pandas as pd
from datetime import datetime
import tags_utils

output_folder = ""

# define the function to extract tags from BAM files
def extract_tags_from_bam(bam_file_path, reg_full_tag, reg_short_tag, output_path, num_threads):
    """Extract samples and tags from a bam file
    """
    cells_tags_to_list = []
    f = 0
    r = 0
    samfile = pysam.AlignmentFile(bam_file_path, "r", threads=num_threads)
    for i, row in enumerate(samfile):
        if (
            not row.has_tag("GN")
            and row.has_tag("CB") 
            and row.has_tag("UB")
            and reg_full_tag.findall(row.seq)
        ):
            
            if row.is_reverse:
                r +=1
            else:
                f +=1

            cell_tag_obj = CellTag.CellTag(i, row, reg_full_tag, reg_short_tag)
            cells_tags_to_list.append(cell_tag_obj.to_list())

    print("forward: %s, reverse: %s" %(f,r))
    return pd.DataFrame(cells_tags_to_list, columns=["row_index", "cell", "umi", "tag"])
    


# create an ArgumentParser object
parser = argparse.ArgumentParser(description='Extract tags from a BAM file.')

# add an argument for the path of the BAM file
parser.add_argument('bam_path', type=str, help='path of the BAM file')

# add an optional argument for the barcode prefix
parser.add_argument('--barcode_prefix', type=str, default='CCGGT',
                    help='prefix of the barcode (default: CCGGT)')

# add an optional argument for the barcode suffix
parser.add_argument('--barcode_suffix', type=str, default='GAATTCG',
                    help='suffix of the barcode (default: GAATTCG)')

parser.add_argument('--num_threads', type=int, default=20,
                    help='number of threads to use (default: 20)')


parser.add_argument('--inner_pattern', type=str, default="[ACGT]{8}",)

parser.add_argument('--allow_seq_error', default=False, action='store_true', help='allow sequence errors in the tags')


parser.add_argument('--output_path', type=str,
                    help='output path for the tags files')

# parse the command line arguments
args = parser.parse_args()

if args.allow_seq_error:
    reg_full_tag_list, reg_short_tag_list = tags_utils.get_regex_with_seq_errors(args.barcode_prefix, args.inner_pattern,args.barcode_suffix)
    
else:
    reg_full_tag = "%s%s%s" % (args.barcode_prefix, args.inner_pattern,args.barcode_suffix)
    reg_short_tag = "%s(%s)%s" % (args.barcode_prefix, args.inner_pattern, args.barcode_suffix)

    reg_full_tag_list = [re.compile(reg_full_tag)]
    reg_short_tag_list = [re.compile(reg_short_tag)]


if not os.path.exists(args.bam_path):
    print("The provided path does not exist.")
    exit()

start_time = datetime.now()
dataframe_list = [] 
for i in range(len(reg_full_tag_list)):
    reg_full_tag = reg_full_tag_list[i]
    reg_short_tag = reg_short_tag_list[i]
    df = extract_tags_from_bam(args.bam_path, reg_full_tag, reg_short_tag, args.output_path, num_threads=args.num_threads)
    dataframe_list.append(df)

dataframe_list = pd.concat(dataframe_list)
dataframe_list = dataframe_list.drop_duplicates()
dataframe_list.to_csv(args.output_path, sep="\t", index=False)
dataframe_list.to_parquet(args.output_path.replace(".csv", ".parquet"), compression=None)

current_time = datetime.now()
print("Finished handling file: %s, this took: %s" %(args.output_path, current_time - start_time))

