beta = [1,2,3]
alpha = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
gamma = [0.3,0.4,0.5,0.6,0.7]

index = 0
fp =open('grid_log','w')
for b in beta:
    for a in alpha:
        for g in gamma:
            index += 1
            fp.write('%.2f,%.2f,%.2f,%d\n'%(b,a,g,index))
            print 'python meteor6.py' + ' -b ' + str(b) + ' -a ' + str(a) + ' -g ' + str(g) + ' > ' + ' q_' + str(index)
            print './grade < ' + ' q_' + str(index) + ' >> score_log'
