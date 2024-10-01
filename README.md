To clone the repository:  
```
git clone https://github.com/CeviKle/MVRC25-starterkit.git 
```
Create conda environment :  

```
conda create -n underwater python=3.8
conda activate underwater
cd MVRC-starterkit 
pip install -r requirements.txt
```

To run demo restoration code for DepthCue [1]
```
cd MVRC-starterkit 
python test.py
```



```[1] Desai, Chaitra, et al. "Depthcue: Restoration of underwater images using monocular depth as a clue." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023.```