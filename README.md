# Super_Resolution_SYS843

Ce dépot contient les codes des méthodes SRCNN, FSRCNN et SRGAN réalisées dans le cadre du projet de super résolution dans le domaine de l'imagerie médicale. 
pour évaluer les performance de ces modèle nous utilisons le Peak Signal Noise Rase (PSNR)


 # SRCNN 
 
 Super Resolution Convolutional Neural Networks
 
 Le modèle srcnn est un modèle de reconstruction à super-résolution d'image unique basé sur un réseau de neurones convolutifs. La structure du modèle est très simple, 
 seules trois couches de structure de réseau neuronal sont utilisées. la structure de ce modèle est illustrée par la figure suivante 
 
 ![image](https://user-images.githubusercontent.com/96759281/147539837-a1143e0d-6712-4e0d-9304-2566f4b5f51c.png)

[[paper](https://arxiv.org/pdf/1501.00092.pdf)][[code](https://github.com/yjn870/SRCNN-pytorch)]

 # implémentation SRCNN
 
 Requirements

    PyTorch 1.0.0
    Numpy 1.15.4
    Pillow 5.4.1
    h5py 2.8.0
    tqdm 4.30.0
    
   1- éxxécuter le fichier prepare.py pour preparer les données préalablement
   2- entrainement 
   #
      python train.py --train-file "train_file/image.h5" \
                --eval-file "eval_file/image.h5" \
                --outputs-dir "image/outputs" \
                --scale 4 \
                --lr 1e-3 \
                --batch-size 16 \
                --num-epochs 200 \
                --num-workers 8 \
                --seed 123    
    3- Test 
    #
        python test.py --weights-file "dir/srcnn.pth" \
                     --image-file "data/img.bmp" \
                     --scale 4
