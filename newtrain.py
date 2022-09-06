import spacy
import random
from spacy.training.example import Example

TRAIN_DATA = [('Con descuentos del 30 por ciento en todos los departamentos, Cuidado con el Perro abri� su segunda tienda en Ciudad Ju�rez ayer.La nueva sucursal de la marca de ropa mexicana se ubica en la planta alta de Las Misiones.Su apertura caus� revuelo entre los juarenses, que abarrotaron la tienda para aprovechar las ofertas de inauguraci�n.""", """La tienda mexicana de ropa Cuidado Con el Perro (CCP) está rematando la tienda en hasta -70 por ciento de descuento en prendas seleccionadas.Y es que con la llegada de las rebajas las tiendas est�n buscando sacar ya toda la mercanc�a para tener nuevo stock.""","""Las dos primeras aperturas de Cuidado con el Perro concretadas en el mes en curso se celebraron los d�as 14 y 26. En una primera instancia, la firma apunt� al norte de M�xico para subir la persiana de una nueva tienda en el estado de Chihuahua, espec�ficamente en Ciudad Cuauht�moc. El establecimiento a pie de calle fue instalado en la colonia centro y se suma a cerca de una decena que la firma operaba en el estado.Poco m�s de una semana despu�s, la marca nacida en 2007 aterriz� en el Golfo de M�xico, para concretar su apertura en Minatitl�n, Veracruz, igualmente con un establecimiento a pie de calle en la zona centro. Seg�n el directorio de Cuidado con el Perro, esta es su primera tienda en dicha ciudad veracruzana.""","""Continuando su serie �Datos curiosos sobre las tiendas�, el tiktoker Alex Pe�aolza afirma que tiendas como C&A y Cuidado con el Perro son de �clase baja�.Los creadores de contenidos siguen posicion�ndose d�a con d�a, dejando de ser una moda para convertirse en una de las profesiones más solicitadas, principalmente, por parte de las nuevas generaciones.Asi lo revela una encuesta firmada por Morning Consult, la cual indica que, tan s�lo en Latinoam�rica, el 86 por ciento de las personas de entre 13 y 38 a�os de edad tienen en mente convertirse en creadores de contenidos de tiempo completo.""","""Muchos de los empleados que cuentan con un esquema de trabajo con salarios y prestaciones, tienen derecho a contar con su tarjeta S� Vale, que es el monedero electr�nico de vales de despensa m�s famoso del pa�s. En Heraldo Binario te decimos cómo funciona este plástico, en qué tienda la aceptan y qué DESCUENTOS te brinda. ""","""l proveedor de soluciones de pago BPC, anunci� una alianza con el emisor de tarjetas y vales, Up Si Vale, para respaldar infraestructura tecnol�gica en pagos. La uni�n beneficiar� a los 4 millones de clientes de la plataforma de pagos.Mediante el programa de Transformacion Digital, Up Si Vale impulsara el desarrollo y eficiencia en su organizaci�n, por lo que seleccion� a la empresa BPC para enrutar sus transacciones y proveer herramientas de prevenci�n del fraude.�Las herramientas de prevenci�n de fraude tienen la capacidad de ver la trama de las transacciones: qui�n y d�nde compr�, la direcci�n, el horario, cantidad de veces intentado. Con esa informaci�n las empresas pueden generar reglas e inteligencia de negocio, que permiten atender situaciones en tiempo real�, coment� Daniel Hern�ndez, director de desarrollo comercial en M�xico de BPC.""","""La firma de comercio electr�nico se�al� que por fin es posible adquirir productos en su sitio web utilizando tarjetas de vales de despensa. \xa0Con este nuevo servicio, m�s de 5 millones de clientes con tarjetas de vales de despensa podr�n pagar en el ecommerce sin mayor problema. \xa0�Qu� tarjetas de vales de despensa son v�lidas? De acuerdo con Amazon México, las tarjetas que serán admitidas serán las de Up Si Vale y Edenred.', {'entities': [(3542, 3549, 'ORG'), (2369, 2376, 'ORG'), (2070, 2077, 'ORG'), (1452, 1472, 'ORG'), (637, 657, 'ORG'), (371, 390, 'ORG'), (61, 81, 'ORG')]})]


def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('es')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        print("no hay ner")
        ner = nlp.create_pipe('ner')
        nlp.add_pipe('ner')
        
    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            

            for batch in spacy.util.minibatch(TRAIN_DATA, size=2):
                for text, annotations in batch:
                    # create Example
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    # Update the model
                    nlp.update([example], losses=losses, drop=0.3)
    return nlp


prdnlp = train_spacy(TRAIN_DATA, 20)

# Save our trained Model
modelfile = input("Enter your Model Name: ")
prdnlp.to_disk(modelfile)

#Test your text
test_text = input("Enter your testing text: ")
doc = prdnlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)