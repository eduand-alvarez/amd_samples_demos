# CUDA to HIP

Instructions: 
1. run setup.sh to enter the container (might need to change some parameters to fit your setup)
2. hipify-perl examine <cuda file>
3. hipify-perl <cuda file> > <hip file>
4. hipcc <hip file> -o <hip program>
