# Machine-Learning-Exercise
Fully Convolutional Neural Network to classify each pixel in an aerial image as roof or non-roof.

# DONE
- [x] Trained on 50x50 image patches -> underfitting
- [x] Trained on 250x250 image patches for 20 epochs --> sparse
- [ ] Train using cross-validation
- [ ] Train on different sizes

# How to use

To run, simply run "run_code.py":

e.g.:

`n_dim = 250`

`weights = 'Weights/weights.h5'`

`test_pred = run_code('image.tif','labels.tif',n_dim,weights,train=True,valid=False)`

Specify:

- n_dim (int specifying the height and width of the image (n by n))
- weights (path to weights file)
- train_image/train_labels. 
- Retrain via `train == True` 
- Train using cross-validation via `valid == True`
