import re
import pysam

class CellTag(object):
    """
    Save and hold the data about the cell barcode, the umi barcode and the tag seen in this cell
    CB - cell barcode
    GN - gene name
    UB - umi barcode
    CR - cell record (without the -1)    
    """
    def __init__(self, row_index:int, aligned_segment: pysam.AlignedSegment, reg_full_tag:re.Pattern, reg_short_tag:re.Pattern):
        self.row_index = row_index
        self.algined_segment = aligned_segment
        self.cell_barcode = aligned_segment.get_tag("CB")
        self.umi_barcode = aligned_segment.get_tag("UB")
        self.full_tag = reg_full_tag.findall(aligned_segment.seq)[0]
        self.small_tag = reg_short_tag.findall(aligned_segment.seq)[0]
        assert len(self.small_tag) == 8
        
    def __lt__(self, other):
        return self.small_tag < other.small_tag
    
    def __eq__(self, other):
        return self.cell_barcode == other.cell_barcode and self.umi_barcode == other.umi_barcode and self.small_tag == other.small_tag
    
    def to_list(self):
        return [self.row_index, self.cell_barcode, self.umi_barcode, self.small_tag]
