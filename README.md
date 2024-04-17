# A Sistema de recomendación de juegos STEAM.
![image](https://github.com/SaulZambranoMeza/Sistema-de-recomendacion-steam-juegos/assets/99093279/36d496e0-7e66-486d-ae3c-0775931203af)



El presente proyecto está dedicado a crear un motor de recomendación de juegos utilizando una base de datos proporcionado por Steam. El objetivo es que mediante una consulta, escribiendo un juego en específico, el modelo pueda recomendar de manera adecuada otros  juegos que pudieran ser de tu agrado.

## A Descripción del proyecto

Iniciar un nuevo rol como Científico de Datos en Steam, una de las principales plataformas multinacionales de videojuegos, promete una emocionante travesía en el mundo del análisis de datos. Sin embargo, el entusiasmo inicial se ve desafiado de inmediato por la cruda realidad: la tarea urgente de crear un **sistema de recomendación de videojuegos** para los usuarios de Steam. Al sumergirte en los datos disponibles, te enfrentas a un panorama desolador: la falta de madurez de los mismos es evidente, con estructuras anidadas y carentes de formato definido. La ausencia total de procesos automatizados para la actualización de nuevos productos. 

![image](https://github.com/SaulZambranoMeza/Sistema-de-recomendacion-steam-juegos/assets/99093279/f687f524-d711-4669-893c-f36458f147fc)


## A Objetivos

Mi trabajo como científico de datos, consiste en afrontar el complejo desafío de desarrollar un sistema de recomendación de videojuegos centrado en el usuario.

Crear un sistema de recomendación de videojuegos basado en datos para Steam, incorporando una API fácil de usar con FastAPI para un acceso perfecto a sugerencias personalizadas.

Realizar un análisis de datos exploratorios (EDA) y entrene un modelo de aprendizaje automático, centrándose en aprovechar complejos algoritmos de similitud de juegos para obtener recomendaciones personalizadas.

Funcionalidad API

El framework FastAPI se utilizó para crear una API que ofreciera puntos finales específicos:

**PlayTimeGenre(genre: str):** identifica el año más popular para un género determinado.
**UserForGenre(genre: str):** busca el usuario principal y su tiempo de reproducción anual para un género específico.
**UsersRecommend(año: int):** enumera los tres juegos más recomendados por los usuarios para un año en particular.
**UsersNotRecommend(año: int):** destaca los tres juegos principales con menos recomendaciones para un año específico.
**sentiment_analysis(year: int):** analiza los sentimientos de las reseñas de los usuarios según el año de lanzamiento del juego.

Aprendizaje automático:

**recomend_game(game_id, top_n=5):** toma un ID de producto como entrada y debe devolver una lista de 5 juegos recomendados que son similares al juego de entrada.

