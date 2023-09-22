import os
import matplotlib.pyplot as plt

## Rerun with align option

directory = "C:/Users/Erik Bjerned/Documents/Git_repos/ABP/L2/raw"
count = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    print(f)
    if os.path.isfile(f):

        f = f.replace("\\", "/")

        data = []
        
        
        with open(f, "r") as file:
            for line in file:
                split_line = line.split()
                data.append(split_line)
            
            file.close()
        #print(data) 
        sizeN_vec = [float(line[4]) for line in data]
        sizeM_vec = [float(line[5]) for line in data]
        mem_vec = [float(line[-2]) for line in data]
        gflops_vec = [float(line[-5]) for line in data]
        styles = ["-", "-", "-", "-", "-", "-", "k--", "-", "k--", "-", "k--"]
        #print(mem_vec)
        plt.xscale("log")
        
        
        if(count == 9 or count == 10):
            plt.plot(sizeM_vec, mem_vec, styles[count], linewidth=1)
        else:
            plt.plot(sizeN_vec, mem_vec, styles[count], linewidth=1)
        
        #plt.axes((512, 1e8, 0, 256))
        if count == 6:
            plt.legend(["32", "64", "128", "256", "512", "1024", "cuBLAS"])
            plt.grid()
            plt.title("Matrix-vector multiplication, M = N")
            plt.ylabel("Memory throughput [GB/s]")
            plt.xlabel("Size N [-]")
            plt.savefig("Matvec.png")
            plt.show()
            plt.clf()
        if count == 8:
            plt.legend(["Implemented", "cuBLAS"])
            plt.grid()
            plt.title("Matrix-vector multiplication, fixed N=10000")
            plt.ylabel("Memory throughput [GB/s]")
            plt.xlabel("Size M [-]")
            plt.savefig("MatvecfN.png")
            plt.show()
            plt.clf()
            
        if count == 10:
            plt.legend(["Implemented", "cuBLAS"])
            plt.grid()
            plt.title("Matrix-vector multiplication, fixed M=16384")
            plt.ylabel("Memory throughput [GB/s]")
            plt.xlabel("Size N [-]")
            plt.savefig("MatvecfN.png")
            plt.show()
            plt.clf()
            
            
            
        
            
            
            
    count += 1