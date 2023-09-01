# SARDINet: Impact of the Decoding Strategy

This repository is dedicated to the impact of the decoding strategy on the optical reconstruction of SARDINet (from [Bralet *et al*, 2022]) from a SAR image. We implemented here five different decoding strategies :

1. Post-upsampling convolutions
2. Transposed convolutions
3. Sub-pixel convolutions from [Shi *et al*, 2016]
4. Post-upsampling convolution with a last sub-pixel convolution layer
5. Transposed convolutions with a last sub-pixel convolution layer

More details can be found in the article [Impact de la stratégie de décodage sur la traduction de modalité radar-optique d'images de télédétection](https://hal.science/hal-04191879)

The code was computed using Python 3.10.6 and the packages versions detailed in the file `requirements.txt`.

In order to run the code please follow the next steps :

* Open the `main.py` file
* Choose the path where your data are located and the path where you want to save the results
* Choose your hyperparameters, the number of the decoder (as mentionned above) and your loss functions
* If you want to modify the architecture of the network, you'll find all you need in the `TransNet.py` file. 
* Save your changes
* Run the `main.py` file

### Citation

If this work was useful for you, please ensure citing our works : 

*Bralet, A., Atto, A., Chanussot, J., and Trouve, E. (2023). Impact de la stratégie de décodage sur la traduction de modalité radar-optique d’images de télédétection. In 29° Colloque sur le traitement du signal et des images, number 2023-1309, pages p. 929–932, Grenoble. GRETSI - Groupe de Recherche en Traitement du Signal et des Images*

Thank you for your support

### Any troubles ?

If you have any troubles with the article or the code, do not hesitate to contact us !

### References

[Bralet *et al*, 2022] A. Bralet, A. M. Atto, J. Chanussot and E. TrouvÉ, "Deep Learning of Radiometrical and Geometrical Sar Distorsions for Image Modality translations," 2022 IEEE International Conference on Image Processing (ICIP), Bordeaux, France, 2022, pp. 1766-1770, doi: 10.1109/ICIP46576.2022.9897713.

[Shi *et al*, 2016] W. Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 1874-1883, doi: 10.1109/CVPR.2016.207.