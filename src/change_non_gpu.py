import os
import subprocess

def makefile_logic():
    makefile = open('Makefile', 'r')
    Lines = makefile.readlines()
    
    for line in Lines[:3]:
        if line.strip() == "GPU=0":
            return

    bashcmd = "sed -i 's/GPU=1/GPU=0/' Makefile"
    output = subprocess.check_output(['bash', '-c', bashcmd])

    bashcmd = "sed -i 's/CUDNN=1/CUDNN=0/' Makefile"
    output = subprocess.check_output(['bash', '-c', bashcmd])

    bashcmd = "sed -i 's/CUDNN_HALF=1/CUDNN_HALF=0/' Makefile"
    output = subprocess.check_output(['bash', '-c', bashcmd])
        
    bashcmd = "make"
    output = subprocess.check_output(['bash', '-c', bashcmd])

if __name__ == '__main__':
    makefile_logic()

