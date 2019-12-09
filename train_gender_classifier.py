from data_loader import gender_load_dataset
from model import GenderModel
from utils.data_augmentation import ImageGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

# parameters
batch_size = 32
num_epochs = 1000
validation_split = .2
do_random_crop = False
patience = 100
num_classes = 2
input_shape = (64, 64, 1)

if input_shape[2] == 1:
    grayscale = True
images_path = './datasets/imdb_crop/'
log_file_path = './models/model1/gender_training.log'
trained_models_path = './model/model1/gender_mini_XCEPTION'
train_keys, val_keys,ground_truth_data = gender_load_dataset()

image_generator = ImageGenerator(ground_truth_data, batch_size,
                                 input_shape[:2],
                                 train_keys, val_keys, None,
                                 path_prefix=images_path,
                                 vertical_flip_probability=0,
                                 grayscale=grayscale,
                                 do_random_crop=do_random_crop)

# model parameters/compilation
model = GenderModel(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# model callbacks
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/2), verbose=1)
csv_logger = CSVLogger(log_file_path, append=False)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# training model
model.fit_generator(image_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_keys) / batch_size),
                    epochs=num_epochs, verbose=1,
                    callbacks=callbacks,
                    validation_data=image_generator.flow('val'),
                    validation_steps=int(len(val_keys) / batch_size))
