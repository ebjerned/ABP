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
        styles = ["-", "-", "-", "-", "k--", "-", "k--", "-", "k--", "-", "-", "-", "k--","-", "-"]
        #print(mem_vec)
        plt.xscale("log")
        if(count == 7 or count == 8):
            plt.plot(sizeM_vec, mem_vec, styles[count], linewidth=1)
        elif(count > 8 and count < 14):
            plt.yscale("log")
            plt.plot(sizeN_vec, gflops_vec, styles[count], linewidth=1)
        else:
            plt.plot(sizeN_vec, mem_vec, styles[count], linewidth=1)
        
        #plt.axes((512, 1e8, 0, 256))
        if count == 4:
            plt.legend(["32", "128", "512", "1024", "cuBLAS"])
            plt.grid()
            plt.title("Matrix-vector multiplication, M = N")
            plt.ylabel("Memory throughput [GB/s]")
            plt.xlabel("Size N [-]")
            plt.savefig("Matvec.png")
            plt.show()
            plt.clf()
        if count == 6:
            plt.legend(["Implemented", "cuBLAS"])
            plt.grid()
            plt.title("Matrix-vector multiplication, fixed N=10000")
            plt.ylabel("Memory throughput [GB/s]")
            plt.xlabel("Size M [-]")
            plt.savefig("MatvecfN.png")
            plt.show()
            plt.clf()
            
        if count == 8:
            plt.legend(["Implemented", "cuBLAS"])
            plt.grid()
            plt.title("Matrix-vector multiplication, fixed M=16384")
            plt.ylabel("Memory throughput [GB/s]")
            plt.xlabel("Size N [-]")
            plt.savefig("MatvecfM.png")
            plt.show()
            plt.clf()
            
        if count == 13:
            plt.legend(["32","128", "512", "cuBLAS", "Naive"])
            plt.grid()
            plt.title("Matrix-matrix multiplication, M=N=K")
            plt.ylabel("Computations [GFLOPS/s]")
            plt.xlabel("Size N [-]")
            plt.savefig("Matmat.png")
            plt.show()
            plt.clf()
            
            
            
        
            
            
            
    count += 1