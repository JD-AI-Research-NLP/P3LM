import os
import sys
import threading

DATASET=sys.argv[1]

#=================glge-coqa=======================
LR=[5e-4, 1e-5]
BATCHSIZE = [128]
# warmups_set = [[100, 200, 300, 400], [500, 600, 700, 800]]
WARMUP = [[500]]
# DEVICES_set = ['0','1','2','3']
PRETRAIN='200G_1990k'
# WEIGHT_DECAY=0.0
WEIGHT_DECAY=0.01
# ADAM_BETAS='\'(0.9,0.98)\''
ADAM_BETAS='\'(0.9,0.999)\''
# CLIP_NORM=0.0
CLIP_NORM=0.1
MAX_SOURCE=512
MAX_TARGET=32
MAX_EPOCH=10
MAX_KEEP_EPOCH=8
DEVICES_set = ['\'0,1,2,3\'']
GPU_NUM = len(DEVICES_set[0].split(','))
MAX_SENTENCE=4

DATASPLIT = 'valid'
CHECKPOINTNUM = [10, 9, 8, 7, 6, 5, 4, 3]
# CHECKPOINTNUM = [1]
# BEAM = [10, 9, 8, 7, 6, 5, 4]
BEAM = [4]
LENPEN=[1.0]
MINLEN=[1]
MAXLEN=[32]

#=================glge-personachat=======================
# LR=[5e-4, 1e-5]
LR=[1e-4]
BATCHSIZE = [128]
# warmups_set = [[100, 200, 300, 400], [500, 600, 700, 800]]
# WARMUP = [[500],[1000]]
WARMUP = [[500]]
# DEVICES_set = ['0','1','2','3']
PRETRAIN='200G_1990k'
# WEIGHT_DECAY=0.0
WEIGHT_DECAY=0.01
# ADAM_BETAS='\'(0.9,0.98)\''
ADAM_BETAS='\'(0.9,0.999)\''
# CLIP_NORM=0.0
CLIP_NORM=0.1
MAX_SOURCE=256
MAX_TARGET=32
MAX_EPOCH=15
MAX_KEEP_EPOCH=10
DEVICES_set = ['\'0,1,2,3\'']
GPU_NUM = len(DEVICES_set[0].split(','))
MAX_SENTENCE=8

DATASPLIT = 'valid'
CHECKPOINTNUM = [13]
# CHECKPOINTNUM = [1]
# BEAM = [10, 9, 8, 7, 6, 5, 4]
BEAM = [8]
LENPEN=[1.8]
# LENPEN=[1.0]
MINLEN=[3]
MAXLEN=[32]

# #=================glge-msqg=======================
# LR=[5e-4, 1e-5]
# BATCHSIZE = [128]
# # warmups_set = [[100, 200, 300, 400], [500, 600, 700, 800]]
# WARMUP = [[1000]]
# # DEVICES_set = ['0','1','2','3']
# PRETRAIN='200G_1990k'
# # WEIGHT_DECAY=0.0
# WEIGHT_DECAY=0.01
# # ADAM_BETAS='\'(0.9,0.98)\''
# ADAM_BETAS='\'(0.9,0.999)\''
# # CLIP_NORM=0.0
# CLIP_NORM=0.1
# MAX_SOURCE=256
# MAX_TARGET=32
# MAX_EPOCH=10
# MAX_KEEP_EPOCH=10
# DEVICES_set = ['\'0,1,2,3\'']
# GPU_NUM = len(DEVICES_set[0].split(','))
# MAX_SENTENCE=8

# DATASPLIT = 'valid'
# CHECKPOINTNUM = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
# # CHECKPOINTNUM = [1]
# # BEAM = [10, 9, 8, 7, 6, 5, 4]
# BEAM = [4]
# LENPEN=[1.0]
# MINLEN=[3]
# MAXLEN=[32]

# #=================glge-msnews=======================
# LR=[5e-4, 1e-5]
# BATCHSIZE = [128]
# # warmups_set = [[100, 200, 300, 400], [500, 600, 700, 800]]
# WARMUP = [[1000]]
# # DEVICES_set = ['0','1','2','3']
# PRETRAIN='200G_1990k'
# # WEIGHT_DECAY=0.0
# WEIGHT_DECAY=0.01
# # ADAM_BETAS='\'(0.9,0.98)\''
# ADAM_BETAS='\'(0.9,0.999)\''
# # CLIP_NORM=0.0
# CLIP_NORM=0.1
# MAX_SOURCE=512
# MAX_TARGET=64
# MAX_EPOCH=10
# MAX_KEEP_EPOCH=8
# DEVICES_set = ['\'0,1,2,3\'']
# GPU_NUM = len(DEVICES_set[0].split(','))
# MAX_SENTENCE=4

# DATASPLIT = 'valid'
# CHECKPOINTNUM = [10, 9, 8, 7, 6, 5, 4, 3]
# # CHECKPOINTNUM = [1]
# # BEAM = [10, 9, 8, 7, 6, 5, 4]
# BEAM = [4]
# LENPEN=[1.0]
# MINLEN=[3]
# MAXLEN=[64]

# #=================glge-xsum=======================
# LR=[1e-4, 1e-5]
LR=[1e-4]
# BATCHSIZE = [128]
BATCHSIZE = [256]
# warmups_set = [[100, 200, 300, 400], [500, 600, 700, 800]]
# WARMUP = [[1000]]
WARMUP = [[500]]
# DEVICES_set = ['0','1','2','3']
PRETRAIN='200G_1990k'
# WEIGHT_DECAY=0.0
WEIGHT_DECAY=0.01
# ADAM_BETAS='\'(0.9,0.98)\''
ADAM_BETAS='\'(0.9,0.999)\''
# CLIP_NORM=0.0
CLIP_NORM=0.1
MAX_SOURCE=512
MAX_TARGET=128
MAX_EPOCH=15
MAX_KEEP_EPOCH=10
DEVICES_set = ['\'0,1,2,3\'']
GPU_NUM = len(DEVICES_set[0].split(','))
MAX_SENTENCE=2

DATASPLIT = 'test'
# CHECKPOINTNUM = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
CHECKPOINTNUM = [12]
# BEAM = [10, 9, 8, 7, 6, 5, 4]
BEAM = [8]
LENPEN=[0.8]
MINLEN=[10]
MAXLEN=[64]

# =================glge-squad=======================
LR=[1e-5]
# BATCHSIZE = [128, 64]
BATCHSIZE = [32]
# warmups_set = [[100, 200, 300, 400], [500, 600, 700, 800]]
# WARMUP = [[1000], [500]]
WARMUP = [[1000]]
# DEVICES_set = ['0','1','2','3']
PRETRAIN='200G_1990k'
# PRETRAIN='200G_17epoch'
# WEIGHT_DECAY=0.0
WEIGHT_DECAY=0.01
# ADAM_BETAS='\'(0.9,0.98)\''
ADAM_BETAS='\'(0.9,0.999)\''
# CLIP_NORM=0.0
CLIP_NORM=0.1
MAX_SOURCE=256
MAX_TARGET=64
MAX_EPOCH=10
MAX_KEEP_EPOCH=10
DEVICES_set = ['\'0,1,2,3\'']
GPU_NUM = len(DEVICES_set[0].split(','))
MAX_SENTENCE=4

DATASPLIT = 'valid'
# DATASPLIT = 'test'
# CHECKPOINTNUM = [10, 9, 8, 7, 6, 5, 4, 3, 2]
# CHECKPOINTNUM = [9, 8, 7, 6, 5, 4, 3, 2, 1]
CHECKPOINTNUM = [6, 5, 4]
# CHECKPOINTNUM = [4]
BEAM = [7, 6, 5, 4]
# BEAM = [7]
LENPEN=[1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]
# LENPEN=[1.5]
MINLEN=[5]
MAXLEN=[32]

# #=================glge-cnndm=======================
LR=[1e-4]
# BATCHSIZE = [128, 64]
BATCHSIZE = [512]
# warmups_set = [[100, 200, 300, 400], [500, 600, 700, 800]]
# WARMUP = [[1000], [500]]
WARMUP = [[1000]]
# DEVICES_set = ['0','1','2','3']
# PRETRAIN='200G_1990k'
PRETRAIN='200G_17epoch'
# WEIGHT_DECAY=0.0
WEIGHT_DECAY=0.01
# ADAM_BETAS='\'(0.9,0.98)\''
ADAM_BETAS='\'(0.9,0.999)\''
# CLIP_NORM=0.0
CLIP_NORM=0.1
MAX_SOURCE=512
MAX_TARGET=128
MAX_EPOCH=15
MAX_KEEP_EPOCH=10
DEVICES_set = ['\'0,1,2,3\'']
GPU_NUM = len(DEVICES_set[0].split(','))
MAX_SENTENCE=2

DATASPLIT = 'valid'
# DATASPLIT = 'test'
CHECKPOINTNUM = [15, 14, 13, 12, 11, 10, 9, 8]
# CHECKPOINTNUM = [9, 8, 7, 6, 5, 4, 3, 2, 1]
# CHECKPOINTNUM = [8, 7, 6, 5, 4, 3, 2, 1]
# CHECKPOINTNUM = [14]
BEAM = [7, 6, 5]
# BEAM = [5]
LENPEN=[1.2,1.4]
# LENPEN=[1.4]
MINLEN=[45]
MAXLEN=[110]

#=================glge-gigaword=======================
# LR=[1e-4]
# BATCHSIZE = [128]
# WARMUP = [[10000]]
# PRETRAIN='200G_1990k'
# WEIGHT_DECAY=0.01
# ADAM_BETAS='\'(0.9,0.999)\''
# CLIP_NORM=0.1
# MAX_SOURCE=128
# MAX_TARGET=32
# MAX_EPOCH=6
# MAX_KEEP_EPOCH=5
# DEVICES_set = ['\'0,1,2,3\'']
# GPU_NUM = len(DEVICES_set[0].split(','))
# MAX_SENTENCE=16

# DATASPLIT = 'valid'
# # DATASPLIT = 'test'
# # CHECKPOINTNUM = [6, 5, 4, 3]
# CHECKPOINTNUM = [6]
# # BEAM = [7, 6, 5, 4]
# BEAM = [5]
# # LENPEN=[0.8,0.9,1.0,1.1,1.2]
# # LENPEN=[1.1]
# LENPEN=[0.9]
# MINLEN=[3]
# MAXLEN=[32]

class myThread(threading.Thread):
    def __init__(self, name, cmd):
        threading.Thread.__init__(self)
        self.name = name
        self.cmd = cmd
    def run(self):
        print(self.name)
        os.system(self.cmd)
        
finetune_log = './log/'+DATASET.lower() + '.log'
with open(finetune_log, 'w') as w:
    max_score = 0.0
    for lr in LR:
        for batchsize in BATCHSIZE:
            for warmups in WARMUP:
                threads = []
#                 for i in range(len(warmups)):
#                     warmup = warmups[i]
#                     DEVICES = DEVICES_set[i]
#                     print("Finetune..." + 'lr=' + str(lr) + ' ' + 'batchsize=' + str(batchsize) + ' ' + 'warmup=' + str(warmup))
#                     params = ['sh', 'finetune_jdnet_glge.sh', str(DATASET), str(lr), str(batchsize), str(warmup), str(PRETRAIN), str(WEIGHT_DECAY), str(ADAM_BETAS), str(CLIP_NORM), str(MAX_SOURCE), str(MAX_TARGET), str(MAX_EPOCH), str(MAX_KEEP_EPOCH), str(DEVICES), str(GPU_NUM), str(MAX_SENTENCE)]
#                     cmd = ' '.join(params)
#                     print(cmd)
#                     t = myThread('thread' + str(DEVICES), cmd)
#                     t.start()
#                     threads.append(t)
#                 for t in threads:
#                     t.join()
                print('!!!!!!!!!!!!!!!! ALL Done!!!!!!!!!!!!!!!!!')
                for warmup in warmups:
                    for epoch in CHECKPOINTNUM:
                        for beam in BEAM: #beam
                            for pelt in LENPEN:
                                pelt = round(pelt, 1)
                                for minLen in MINLEN:
                                    for maxLen in MAXLEN:
                                        TEST_BATCHSIZE=128 if beam < 8 else 16
                                        print("epoch" + str(epoch) + "--beam" + str(beam) + "--penal" + str(pelt) + "--minLen" + str(minLen) + "--maxLen" + str(maxLen))
                                        log = 'tmp.' + DATASET.lower() + '.txt'
                                        params = ['sh', 'inference_eva_jdnet_glge.sh', str(DATASET), str(DATASPLIT), str(epoch), str(beam), str(pelt), str(minLen), str(maxLen), str(MAX_SOURCE), str(MAX_TARGET), str(TEST_BATCHSIZE), str(lr), str(batchsize), str(warmup), str(PRETRAIN), str(WEIGHT_DECAY), str(ADAM_BETAS), str(CLIP_NORM), str(log)]
                                        cmd = ' '.join(params)
                                        print(cmd)
                                        os.system(cmd)
                                        with open(log, 'r') as f:
                                            scores_str = f.readline().strip().split('/')
                                            scores = [float(s) for s in scores_str]
                                            score = sum(scores)/len(scores)
                                            if score > max_score:
                                                max_score = score
                                                setting = {'epoch': str(epoch),
                                                           'beam': str(beam),
                                                           'pelt': str(pelt),
                                                           'minLen': str(minLen), 
                                                           'maxLen': str(maxLen),
                                                           'MAX_SOURCE': str(MAX_SOURCE),
                                                           'MAX_TARGET': str(MAX_TARGET),
                                                           'TEST_BATCHSIZE': str(TEST_BATCHSIZE),
                                                           'lr': str(lr),
                                                           'batchsize': str(batchsize),
                                                           'warmup': str(warmup),
                                                           'PRETRAIN': str(PRETRAIN),
                                                           'WEIGHT_DECAY': str(WEIGHT_DECAY),
                                                           'ADAM_BETAS': str(ADAM_BETAS),
                                                           'CLIP_NORM': str(CLIP_NORM)}
                                                w.write('Max_score updated: ' + str(max_score) + '(' + ','.join(scores_str) + '): ' + ','.join([k + ':' + v for k, v in setting.items()]) + '\n')
                                                w.flush()
            
            

