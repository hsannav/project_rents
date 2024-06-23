from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame(columns=['Anuncio', 'Precio', 'Inmobiliaria', 'Habitaciones', 'Planta', 'Superficie', 'Exterior', 'Ascensor', 'Tipo', 'Ubic', 'Barrio'])

for web in range(1, 74):
    with open(f'webs/{str(web).zfill(2)}.html', encoding='utf8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        articles = soup.find_all('article')
        anuncios = []
        prices = []
        infos = [[] for _ in articles]
        inmobiliarias = []
        
        for i, article in enumerate(articles):
            try:
                anuncio = article.find('a', {'class': 'item-link', 'role': 'heading'}).get_text().strip()
                anuncios.append(anuncio)

                try:
                    inmobiliaria = article.find('picture', {'class': 'logo-branding'}).find('a').get('title')
                except:
                    inmobiliaria = ''
                inmobiliarias.append(inmobiliaria)

                price = article.find('span', {'class': 'item-price h2-simulated'}).text
                prices.append(price)

                info = article.find_all('span', {'class': 'item-detail'})
                for j in info:
                    infos[i].append(j.text)
            except:
                continue

        while [] in infos:
            infos.remove([])
            
        for k in range(len(anuncios)):
            anadir = [anuncios[k], int(prices[k].replace('€/mes', '').replace('.', '')), str(inmobiliarias[k])]
            habitaciones = np.nan
            superficie = np.nan
            planta = np.nan
            exterior = np.nan
            ascensor = np.nan
            ubic = ' '.join(anuncios[k].split()[2:])
            
            try:
                if ubic.split(',')[-1].strip() == 'València':
                    barrio = ubic.split(',')[-2].strip()
                else:
                    barrio = ubic.split(',')[-1].strip()
            except:
                barrio = np.nan
                
            tipo = anuncios[k].split()[0]
            
            for dato in infos[k]:
                if 'hab' in dato:
                    habitaciones = int(dato.split()[0])
                    
                if 'm' in dato and 'minutos' not in dato:
                    try:
                        superficie = int(dato.split()[0])
                    except:
                        pass
                    
                if 'Planta' in dato:
                    try:
                        planta = int(dato.split()[1].replace('ª', ''))
                    except:
                        pass
                    
                    if 'con' in dato:
                        ascensor = 1
                    if 'sin' in dato:
                        ascensor = 0
                        
                    if 'exterior' in dato:
                        exterior = 1
                    if 'interior' in dato:
                        exterior = 0
                        
            anadir.append(habitaciones)
            anadir.append(planta)
            anadir.append(superficie)
            anadir.append(exterior)
            anadir.append(ascensor)
            anadir.append(tipo)
            anadir.append(ubic)
            anadir.append(barrio)
            
            df.loc[len(df)] = anadir

df.to_excel('pisos.xlsx')