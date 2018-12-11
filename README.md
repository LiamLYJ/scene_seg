# scene_seg
pytorch implementation of scene_seg

pytorch >= 0.4


based on paper: 
One-shot Texture Segmentation
https://arxiv.org/pdf/1807.02654.pdf

Train the model: 
python train.py
(with args settings)

couple results are given in save_dtd(dtd dataset, http://www.robots.ox.ac.uk/~vgg/data/dtd/),  
                            save_scene (use js script to download training dataset)
                            and in real_test(real test results)

Test the model: 
python test.py
(with args setting)

![input image](./real_test/input_img_00.png)
![input texture](./real_test/texture_02.png)
![output mask](./save_dtd/output_02.png)

![input image](./save_dtd/input_img_01.png)
![input texture](./save_dtd/texture_01.png)
![output mask](./save_dtd/output_01.png)

![input image](./save_scene/input_img_01.png)
![input texture](./save_scene/texture_01.png)
![output mask](./save_scene/output_01.png)


                        

