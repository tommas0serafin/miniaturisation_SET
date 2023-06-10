import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Flatten, Dense, Concatenate, GlobalAveragePooling2D, add
from tensorflow.keras.models import Model
import tensorflow_model_optimization as tfmot
import tempfile
import numpy as np

def create_cnn():
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(32, 32, 3)),
        keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def dense_block(x, num_layers, growth_rate):
    for i in range(num_layers):
        # BN - ReLU - Conv(1x1) - BN - ReLU - Conv(3x3)
        x1 = BatchNormalization()(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(growth_rate, (1, 1), padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(growth_rate, (3, 3), padding='same')(x1)
        
        # Concatenate the output with the input
        x = Concatenate()([x, x1])
        
    return x

def transition_layer(x, compression_factor):
    # BN - Conv(1x1) - AveragePooling(2x2)
    x = BatchNormalization()(x)
    x = Conv2D(int(tf.keras.backend.int_shape(x)[-1] * compression_factor), (1, 1), padding='same')(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

def create_densenet(input_shape, num_classes, num_layers_per_block, growth_rate, compression_factor):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Conv(3x3) with 16 filters, stride=1
    x = Conv2D(16, (3, 3), padding='same', strides=(1, 1))(inputs)
    
    # Dense blocks with transition layers
    for i, num_layers in enumerate(num_layers_per_block):
        x = dense_block(x, num_layers, growth_rate)
        if i != len(num_layers_per_block) - 1:
            x = transition_layer(x, compression_factor)
    
    # Global average pooling and fully connected layer
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((2, 2))(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1), activation='relu'):
    y = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation(activation)(y)
    y = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same')(y)
    y = BatchNormalization()(y)

    if strides != (1, 1) or x.shape[-1] != filters:
        x = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(x)
        x = BatchNormalization()(x)

    y = add([x, y])
    y = Activation(activation)(y)
    return y

def create_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = residual_block(x, filters=8, strides=(2, 2))
    x = residual_block(x, filters=8, strides=(1, 1))
    x = residual_block(x, filters=8, strides=(1, 1))

    x = residual_block(x, filters=16, strides=(2, 2))
    x = residual_block(x, filters=16, strides=(1, 1))
    x = residual_block(x, filters=16, strides=(1, 1))

    x = residual_block(x, filters=32, strides=(2, 2))
    x = residual_block(x, filters=32, strides=(1, 1))
    x = residual_block(x, filters=32, strides=(1, 1))

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def prune_model(model, target_sparsity):
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
            target_sparsity=target_sparsity,
            begin_step=0,
            end_step=-1,
            frequency=1
        )
    }

    model_pruned = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model_pruned.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])

    return model_pruned

import time
import numpy as np
import tensorflow as tf

def evaluate_model(interpreter, data_generator):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image generated by the data generator.
    prediction_probs = []
    test_labels = []
    inference_times = []
    for i in range(len(data_generator)):
        batch_images, batch_labels = data_generator[i]
        batch_size = batch_images.shape[0]

        # Pre-processing: convert to float32 to match with the model's input data format.
        batch_images = batch_images.astype(np.float32)

        # Run inference on each image in the batch and measure the inference time.
        for j in range(batch_size):
            image = batch_images[j:j+1]
            start_time = time.perf_counter()
            interpreter.set_tensor(input_index, image)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            probs = output.squeeze()
            prediction_probs.append(probs)
            test_labels.append(batch_labels[j])

    # Convert probabilities to binary predictions.
    prediction_probs = np.array(prediction_probs)
    prediction_binary = (prediction_probs > 0.5).astype(int)

    # Calculate accuracy, precision, and recall.
    test_labels = np.array(test_labels)
    confusion_matrix = tf.math.confusion_matrix(test_labels, prediction_binary)
    true_positives = confusion_matrix[1][1]
    false_positives = confusion_matrix[0][1]
    false_negatives = confusion_matrix[1][0]
    accuracy = (prediction_binary == test_labels).mean()
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    # Calculate the average inference time per frame.
    inference_time_per_frame = sum(inference_times) / len(inference_times)

    return accuracy, precision.numpy(), recall.numpy(), inference_time_per_frame


def main():
    # Choose the model
    model_choice = input("Choose the model (1 for CNN, 2 for DenseNet, 3 for ResNet): ")

    if model_choice == '1':
        model = create_cnn()
    elif model_choice == '2':
        model = create_densenet(
            input_shape=(32, 32, 3),
            num_classes=1,
            num_layers_per_block=[2, 2, 2],
            growth_rate=16,
            compression_factor=0.5
        )
    elif model_choice == '3':
        model = create_resnet(input_shape=(32, 32, 3), num_classes=1)
    else:
        print("Invalid choice. Please choose 1, 2, or 3.")
        return

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Load and preprocess the data
    train_dir = 'your own directory'

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(32, 32),
        batch_size=16,
        class_mode='binary',
        shuffle=True,
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(32, 32),
        batch_size=16,
        class_mode='binary',
        shuffle=False,
        subset='validation'
    )

    # Train the model with desired sparsity
    sparsity = float(input("Enter the desired sparsity (e.g., 0.70): "))

    if sparsity == 0:
        # No pruning callbacks needed
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size
        )
    else:
        model_pruned = prune_model(model, target_sparsity=sparsity)
        pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
        callbacks = [pruning_callback]

        model_pruned.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=2,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            callbacks=callbacks
        )
        model = tfmot.sparsity.keras.strip_pruning(model_pruned)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    _, tflite_file = tempfile.mkstemp('.tflite')

    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)

    print('Saved TFLite model to:', tflite_file)

    # Convert to quantized TFLite
    quantize_model = input("Quantize the model? (y/n): ")

    if quantize_model.lower() == 'y':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quantized_model = converter.convert()

        _, quantized_file = tempfile.mkstemp('.tflite')

        with open(quantized_file, 'wb') as f:
            f.write(tflite_quantized_model)

        print('Saved quantized TFLite model to:', quantized_file)
    
    test_dir = 'your own directory'
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(32, 32),
        batch_size=16,
        class_mode='binary',
        shuffle=False
    )

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    accuracy, precision, recall, inference_time = evaluate_model(interpreter, test_generator)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Inference Time per Frame:", inference_time)

if __name__ == '__main__':
    main()
