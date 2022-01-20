from src.gan.gan import Gan
from src.utils.data_provider import DataProvider, DataType
from src.generator.generator import Generator, GeneratorType
from src.discriminator.discriminator import Discriminator, DiscriminatorType

data_provider = DataProvider(DataType.ALL, show=False, cut=10)
generator = Generator(GeneratorType.SMALL)
discriminator = Discriminator(DiscriminatorType.MEDIUM)

main_gan = Gan(generator, discriminator, data_provider)

if __name__ == '__main__':
    main_gan.load_data()
    main_gan.train()
    main_gan.make_gif()
