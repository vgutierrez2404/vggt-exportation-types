import subprocess
import datetime
from timezonefinder import TimezoneFinder
import pytz

from moviepy.video.io.VideoFileClip import VideoFileClip
from dateutil import tz
from dateutil import parser
import json

"""WARNING - THESE IS THE SAME FUNCTION AS IN ClenaPointClouds/dateUtilsForFrames

ALL THE CODE MODIFICATED HERE SHOULD BE REPLICATED IN THE ClenaPointClouds/dateUtilsForFrames 
FILE. 

NOT DONE YET!!! @VICTOR
"""

def datetime_to_timestamp_utc(dt):
    """
    Converts datetime variable into unix timestamp https://en.wikipedia.org/wiki/Unix_time
    """
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    timestamp = (dt - epoch).total_seconds() * 10 ** 9

    return int(timestamp)

# FUNCIONA
def get_modification_datetime_UTF(file_path:str) -> None:
    """
    This function modificates the date of the file in the system and converts 
    it to a datetime with UTC reference. 
    
    Input: 
        - file_path: string containing the path of the file .

    Output: 
        - modification_datetime_utc
    """
    command = ["stat", "-c", "%y", file_path]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0: # tendría el mismo efecto con un try-catch
        modification_str = result.stdout.strip()
        parsed_datetime = parser.parse(modification_str)
        local_timezone = tz.tzlocal()
        local_datetime = parsed_datetime.replace(tzinfo=local_timezone)
        utc_timezone = tz.UTC
        modification_datetime_utc = local_datetime.astimezone(utc_timezone).replace(tzinfo=datetime.timezone.utc)

        return modification_datetime_utc
    
    else:
        # Error al ejecutar el comando stat
        print("Error:", result.stderr)
        return None
    
    # We should change the code with this below, but first we must ensure it works 
    # properly. 
    """
    try:
        command = ["stat", "-c", "%y", file_path]
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Parse the modification datetime string
        modification_str = result.stdout.strip()
        parsed_datetime = parser.parse(modification_str)

        # Local timezone
        local_timezone = tz.tzlocal()
        local_datetime = parsed_datetime.replace(tzinfo=local_timezone)

        # Convert to UTC
        utc_timezone = tz.UTC
        modification_datetime_utc = local_datetime.astimezone(utc_timezone).replace(tzinfo=datetime.timezone.utc)

        return modification_datetime_utc
    
    except subprocess.CalledProcessError as e:
        # Error executing the subprocess command
        print("Error executing stat command:", e.stderr)
        return None

    except (ValueError, parser.ParserError) as e:
        # Error parsing the datetime or incorrect datetime format
        print("Error parsing the modification datetime:", e)
        return None

    except Exception as e:
        # General exception handling for any other unexpected errors
        print("An unexpected error occurred:", e)
        return None
    """

# No funciona
def get_video_creation_date(filename):
    try:
        clip = VideoFileClip(filename)
        metadata = clip.reader.metadata
        creation_date_str = metadata.get('creation_time', None)
        clip.reader.close()

        if creation_date_str:
            creation_date = datetime.strptime(creation_date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            return creation_date
        else:
            print("No creation date found in the metadata.")
    except Exception as e:
        print("An error occurred while reading the file:", e)

    return None


def convert_to_utc(latitude, longitude, exif_date):
    """
    Donde se llama a esto? Por lo visto no se llama en ningun script. 

    """
    # Obtener la fecha y hora actual en formato UTC
    utc_now = datetime.datetime.utcnow()

    # Obtener el desplazamiento de tiempo en segundos para la posición GPS
    time_offset = (longitude // 15) * 3600

    # Convertir la fecha devuelta por exiftool a un objeto de datetime
    exif_date_obj = datetime.datetime.strptime(exif_date, "%Y:%m:%d %H:%M:%S")

    # Calcular la fecha en formato UTC ajustando el desplazamiento de tiempo
    utc_date = exif_date_obj - datetime.timedelta(seconds=time_offset)

    # Obtener la fecha en formato de 19 dígitos (sin microsegundos)
    utc_date_19_digits = utc_date.strftime("%Y%m%d%H%M%S")

    return utc_date_19_digits


def convert_time_stam_utc_to_local(timestamp_utc, latitude, longitude):
    """
    Esto tampoco se llama en ningun script exterior
    """
    # Obtener la zona horaria correspondiente a la latitud y longitud GPS
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=latitude, lng=longitude)

    # Crear un objeto datetime en UTC a partir del timestamp
    utc_datetime = datetime.datetime.utcfromtimestamp(timestamp_utc / 1e9).replace(tzinfo=pytz.UTC)

    # Obtener la zona horaria local basada en las coordenadas GPS
    local_timezone = pytz.timezone(timezone_str)

    # Convertir el datetime de UTC a la hora local
    local_datetime = utc_datetime.astimezone(local_timezone)

    return local_datetime
    

def get_video_duration_ffprobe(file_path):
    """
    Esto tampoco se llama en ningun script 
    """
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
               "default=noprint_wrappers=1:nokey=1", file_path]
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode('utf-8').strip()
        duration = float(output)
        return duration
    
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el comando: {e}")
    
    return None


def get_creation_datetime(video_path: str, video_ref=None) -> int:
    """
    This function gets the creation time of the video passed as argument and 
    retures its time in utc value. 

    Input: 
        - video_path: string with the path of the video. 
    
    Output: 
        - creation_datetime_utc: integer.
    """
    # Other alternative:
    #command = ["exiftool", "-createdate", "-s", "-s", "-s", "-DateTimeOriginal", video_path]
    if video_ref:
        command = f"ffprobe -v quiet -select_streams v:0  -show_entries stream_tags=creation_time -of default=noprint_wrappers=1:nokey=1 {video_ref}"
    else:
        command = f"ffprobe -v quiet -select_streams v:0  -show_entries stream_tags=creation_time -of default=noprint_wrappers=1:nokey=1 {video_path}"

    output = subprocess.check_output(command, shell=True, text=True)
    creation_time_str = output.strip()

    if len(creation_time_str) == 0:
        creation_datetime = datetime.datetime.strptime("2023-06-30T19:12:52.000000Z","%Y-%m-%dT%H:%M:%S.%fZ")

    else:
        creation_datetime = datetime.datetime.strptime(creation_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")

    # Returned as UTC
    creation_datetime_utc = creation_datetime.replace(tzinfo=datetime.timezone.utc)

    return creation_datetime_utc


def get_video_creation_date_json(video_file):
    ffprobe_cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_entries', 'format_tags=creation_time',
        video_file
    ]

    try:
        output = subprocess.check_output(ffprobe_cmd, stderr=subprocess.STDOUT)
        json_output = json.loads(output)
        creation_time = json_output['format']['tags']['creation_time']
        return creation_time
    except subprocess.CalledProcessError as e:
        print(f"Error executing ffprobe: {e.output.decode()}")
        return None


def get_modification_datetime(file_path):
    # Ejecutar el comando stat para obtener los metadatos del archivo
    command = ["stat", "-c", "%y", file_path]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        # Obtener la fecha y hora de modificación del resultado
        modification_str = result.stdout.strip()
        modification_datetime = parser.parse(modification_str)
        return modification_datetime
    else:
        # Error al ejecutar el comando stat
        print("Error:", result.stderr)
        return None


