#!/usr/bin/env python

import sys
sys.path.append('../../')
from logparser.Drain import LogParser

#==========================================HDFS
# This part is for HDFS dataset includes : Input dir, output dir, name of log file, log format(all formats in Benchmark file).
# input_dir  = '../dataset/HDFS/' # The input directory of log file
# output_dir = '../dataset/HDFS/'  # The output directory of parsing results
# log_file   = 'HDFS.log'  # The input log file name
# log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
# Regular expression list for optional preprocessing (default: [])
# regex      = [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"]
# st         = 0.5  # Similarity threshold
# depth      = 4  # Depth of all leaf nodes
 
#==========================================BGL
# This part is for BGL dataset includes : Input dir, output dir, name of log file, log format(all formats in Benchmark file).
input_dir  = '../dataset/BGL/' # The input directory of log file
output_dir = '../dataset/BGL/'  # The output directory of parsing results
log_file   = 'BGL.log'  # The input log file name
log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'  # BGL log format
# Regular expression list for optional preprocessing (default: [])
regex      = [r"core\.\d+"],
st         = 0.5  # Similarity threshold
depth      = 4  # Depth of all leaf nodes

#==========================================Call function 
# Call function LogParser 
parser = LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)
