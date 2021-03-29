class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Input(shape=(256, 256, 3)),
          #layers.Conv2D(72, (5,5), activation='relu', padding='same', strides=2),
          layers.Conv2D(52, (5,5), activation='relu', padding='same', strides=2),
          layers.Conv2D(32, (5,5), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
          layers.Conv2DTranspose(32, kernel_size=5, strides=2, activation='relu', padding='same'),
          layers.Conv2DTranspose(12, kernel_size=5, strides=2, activation='relu', padding='same'),
          #layers.Conv2DTranspose(22, kernel_size=5, strides=2, activation='relu', padding='same'),
          layers.Conv2D(3, kernel_size=(5,5), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded