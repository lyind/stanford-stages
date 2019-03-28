from struct import Struct
import numpy as np


def transform_anl_aasm(tuples):
    '''Transform network output (tuple of 5 values) to an iterator over scalar values indicating the sleep stage.'''
    tuples_reordered = np.take(tuples, [3,2,1,4,0], axis=1)
    return np.nditer(np.argmax(tuples_reordered, axis=1))


def serialize_anl(out_file, stages, starttime_seconds, epoch_duration_millies, colors=None, lineWidth=None):
    '''Serialize the values from the 1D generator stages into an ANL file for the specified recording start time.

       Colors may be specified in a list, where the index should match the associated value in stages.
       Colors are specified as little-endian RGB (without alpha).
       
       Line widths (replace event duration for sleep stage files only) may be specified in percent of graph height).
    '''
    
    # use default colorscheme if none has been specified
    if colors is None:
        blue = 0x00FF0000
        red = 0x000000FF
        colors = [red, red, red, blue, red, red]

    # use line width
    if lineWidth is None:
        width_normal = 0x00000000
        width_30percent = 0x1 * 1000 * 1000 * 60 * 60 * 24 * 30  # 30 days: 30% line width
        colors = [width_normal, width_normal, width_normal, width_30percent, width_normal, width_normal]
        
    # convert date from UNIX timestamps to "borland/excel days since 01-01-1900"
    # 25569 days between 01-01-1900 and 01-01-1970
    # 86400 seconds per day
    start_timestamp_us = (starttime_seconds + (25569 * 86400)) * 1000 * 1000
    epoch_duration_us = int(epoch_duration_millies * 1000)

    # serialize header
    out_file.write(b'000000CB\r\n')

    # serialize records for all values that the generator yields
    serializer = Struct('<qqIiiB')
    timestamp_us = int(start_timestamp_us)
    for epoch_stage in stages:
        stage = int(epoch_stage)
        out_file.write(serializer.pack(timestamp_us, lineWidth[stage], colors[stage], 0, stage, 0))
        timestamp_us = timestamp_us + epoch_duration_us

