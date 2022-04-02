docker run --runtime nvidia --rm -it --workdir /home/jupyter -v ${PWD}:/home/jupyter -u jupyter kaggle/python-gpu-build /bin/bash
