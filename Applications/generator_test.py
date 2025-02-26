def build_generator(noise_dim=100):
    noise_input = layers.Input(shape=(noise_dim,))
    label_input = layers.Input(shape=(1,), dtype='int32')
    
    # Embed the label and flatten
    label_embedding = layers.Embedding(num_classes, noise_dim, input_length=1)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    
    # Concatenate noise and label embedding
    merged_input = layers.Concatenate()([noise_input, label_embedding])
    
    x = layers.Dense(7 * 7 * 128, use_bias=False)(merged_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((7, 7, 128))(x)
    
    x = layers.Conv2DTranspose(64, kernel_size=5, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', use_bias=False, activation='tanh')(x)
    
    generator = models.Model([noise_input, label_input], x, name="generator")
    return generator
