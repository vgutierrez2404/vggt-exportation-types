# Siguientes pasos
Una vez tengas instalado el repositorio (te recomiendo que uses un conda environment) puedes hacer una prueba con el ejemplo que ponen ellos. 
Te subo dos pequeños datasets de una y diez imagenes de la misma pila de material (para que veas un ejemplo con el que trabajamos). De todas formas, tienes los ejemplos de los creadores del modelo en vggt/examples. 

Yo tengo una adaptacion del script vggt_to_colmap.py pero exportando la nube una nube de puntos en
formato .ply para que la visualización fuera más simple. 

Te explico un poco los "input arguments" que tiene para que puedas probar. 

### Parametros script 

--input_dir: directorio donde estan tus imágenes 
--output_dir: directorio de resultados. Si corres más de una reconstruccion para el mismo conjunto de imágenes se te harán varias carpetas con los resultados de cada una. 
--conf_threshold: todos los puntos de la nube inferida tienen un valor de confianza que va del 1-100. Este valor significa lo "seguro" que está el modelo de que el punto esté representado correctamente. El script usa el valor que indicas en conf_threshold para filtrar de la nube de puntos los valores con menor confianza que la indicada en el hiperparámetro. 
--mask_sky: filtra los puntos que un modelo asume que están asociados al cielo. 
--mask_black_bg: filtro básico para quitar los puntos negros.
--mask_white_bg: como el anterior pero con puntos blancos. 
--export_ply: si quieres exportar el modelo en formato .ply. Esto te ayuda en caso en que tengas instalado software para visualizar nubes de puntos (te recomiendo meshlab). Si es así puedes abrirlo en dicho programa.
--custom_frame_selection: puedes pasarle una list de frames de tu directorio para que el modelo utilize esos en vez de todos los contenidos en el directorio 

Los demas parámetros no son relevantes por el momento. Tienes sus descriptciones en el main() por si te interesa. 

### Ejemplo 1 

Para correr un reconstruccion solo tienes que hacer: 

1) Activar tu conda environment (o tu virtual environment) una vez instalados los requirements. 

2) Correr el script desde el directorio padre: 

```python 
python3  vggt_to_ply.py --image_dir directorio/de/imagenes --output_dir directorio/de/imagenes/output_files --conf_threshold 50 --export_ply
```
Puedes jugar con los parámetros mencionados anteriormente. 

3) Deberías tener un archivo points3D.ply con tu reconstrucción. Veras que tenemos mucho margen de mejora. 
Ademas, hay un archivo de log que muestra un poco de información de la reconstrucción, lo utilizaremos más adelante. 