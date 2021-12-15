from src.gan_provider import GANProvider
from src.data_provider import DataProvider, DataType
from src.generator_provider import GeneratorProvider, GeneratorType
from src.discriminator_provider import DiscriminatorProvider, DiscriminatorType

data_provider = DataProvider(DataType.ALL, show = False, cut = 10)
generator_provider = GeneratorProvider(GeneratorType.SMALL)
discriminator_provider = DiscriminatorProvider(DiscriminatorType.MEDIUM)

gan_provider = GANProvider(generator_provider, 
                           discriminator_provider, 
                           data_provider)

gan_provider.load_data()
gan_provider.train()
gan_provider.make_gif()