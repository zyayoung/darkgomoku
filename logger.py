from tensorboardX import SummaryWriter
writer = SummaryWriter(flush_secs=1)
while True:
    writer.add_scalar(input(), float(input()), int(input()))
