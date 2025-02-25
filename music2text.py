import streamlit as st
import librosa
import librosa.display
import numpy as np
import openai
import json
from keyfinder import Tonal_Fragment
from pydub import AudioSegment
import os
import logging
import tempfile
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import StandardScaler
from transformers import pipeline  # For the HF pipeline
import torch
import math
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set your OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("No OPENAI_API_KEY environment variable found.")

# DigitalOcean Spaces URLs for your models
MODEL_URL = "https://music2text-models.nyc3.digitaloceanspaces.com/genres_classification/model.safetensors"
PYTORCH_URL = "https://music2text-models.nyc3.digitaloceanspaces.com/genres_classification/pytorch_model.bin"

def download_file(url, local_filename):
    if os.path.exists(local_filename):
        print(f"{local_filename} already exists, skipping download.")
        return
    print(f"Downloading {local_filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

def load_models():
    target_dir = "music_genres_classification"
    os.makedirs(target_dir, exist_ok=True)
    download_file(MODEL_URL, os.path.join(target_dir, "model.safetensors"))
    download_file(PYTORCH_URL, os.path.join(target_dir, "pytorch_model.bin"))

def safe_float(value):
    if isinstance(value, (np.ndarray, np.generic)) and value.size == 1:
        return value.item()
    return float(value)

# ==========================================================
# Custom Micro-Genre Generation Functions
# ==========================================================
# (These rules can later be extended or even loaded from a JSON file to help the system learn sub-genres.)
custom_genre_rules = [
    {"conditions": {"pop": 0.7, "electronic": 0.1}, "genre": "Electropop"},
    {"conditions": {"pop": 0.7, "hiphop": 0.1}, "genre": "PopRap"},
    {"conditions": {"rock": 0.6, "metal": 0.3}, "genre": "Heavy Rock"},
    {"conditions": {"jazz": 0.5, "blues": 0.3}, "genre": "JazzBlues"},
    {"conditions": {"rnb": 0.4, "hiphop": 0.3}, "genre": "RnbHop"}
]

def compute_normalized_top_genres(probabilities, top_n=4):
    sorted_genres = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_genres = dict(sorted_genres[:top_n])
    total_score = sum(top_genres.values())
    if total_score > 0:
        normalized = {genre: score / total_score for genre, score in top_genres.items()}
    else:
        normalized = top_genres
    return normalized

def determine_micro_genre(probabilities):
    for rule in custom_genre_rules:
        if all(probabilities.get(genre, 0) >= threshold for genre, threshold in rule["conditions"].items()):
            return rule["genre"]
    return "Undefined"
# ==========================================================
# End Custom Micro-Genre Generation Functions
# ==========================================================

# List of funky genre database to be used by ChatGPT for creative micro-genre naming
FUNKY_GENRES = """
Astro Acid Breaks
Cosmic 8-Bit Garage
Nebula Nu Metal
Stellar Synthwave Dub
Galactic Glitch Punk
Celestial Electro Acid
Orbital Basscore
Lunar Lo-Fi Chillhop
Meteoric Electro Swing
Quantum Industrial Jazz
Solar Flare Future Funk
Interstellar Space Disco
Black Hole Ambient
Cosmic Post-Rock Rumba
Gravity-Defying Techno
Stardust Progressive House
Eclipse Punk Dub
Planetary Acid Jazz
Comet-Driven Neo-Psychedelia
Starship Noisecore
Galactic Acid Breaks
Nebular Drone Fusion
Celestial Bass & Beats
Interplanetary Vaporwave
Cosmic Darkwave Country
2 tone
2-step garage
4-beat
4x4 garage
8-bit
acapella
acid
acid breaks
acid house
acid jazz
acid rock
acoustic music
acousticana
adult contemporary music
african popular music
african rumba
afrobeat
aleatoric music
alternative country
alternative dance
alternative hip hop
alternative metal
alternative rock
ambient
ambient house
ambient music
americana
anarcho punk
anime music
anti-folk
apala
ape haters
arab pop
arabesque
arabic pop
argentine rock
ars antiqua
ars nova
art punk
art rock
ashiq
asian american jazz
australian country music
australian hip hop
australian pub rock
austropop
avant-garde
avant-garde jazz
avant-garde metal
avant-garde music
axé
bac-bal
bachata
background music
baggy
baila
baile funk
baisha xiyue
baithak gana
baião
bajourou
bakersfield sound
bakou
bakshy
bal-musette
balakadri
balinese gamelan
balkan pop
ballad
ballata
ballet
bamboo band
bambuco
banda
bangsawan
bantowbol
barbershop music
barndance
baroque music
baroque pop
bass music
batcave
batucada
batuco
batá-rumba
beach music
beat
beatboxing
beautiful music
bebop
beiguan
bel canto
bend-skin
benga
berlin school of electronic music
bhajan
bhangra
bhangra-wine
bhangragga
bhangramuffin
big band
big band music
big beat
biguine
bihu
bikutsi
biomusic
bitcore
bitpop
black metal
blackened death metal
blue-eyed soul
bluegrass
blues
blues ballad
blues-rock
boogie
boogie woogie
boogie-woogie
bossa nova
brass band
brazilian funk
brazilian jazz
breakbeat
breakbeat hardcore
breakcore
breton music
brill building pop
britfunk
british blues
british invasion
britpop
broken beat
brown-eyed soul
brukdown
brutal death metal
bubblegum dance
bubblegum pop
bulerias
bumba-meu-boi
bunraku
burger-highlife
burgundian school
byzantine chant
ca din tulnic
ca pe lunca
ca trù
cabaret
cadence
cadence rampa
cadence-lypso
café-aman
cai luong
cajun music
cakewalk
calenda
calentanos
calgia
calypso
calypso jazz
calypso-style baila
campursari
canatronic
candombe
canon
canrock
cantata
cante chico
cante jondo
canterbury scene
cantiga
cantique
cantiñas
canto livre
canto nuevo
canto popular
cantopop
canzone napoletana
cape jazz
capoeira music
caracoles
carceleras
cardas
cardiowave
carimbó
cariso
carnatic music
carol
cartageneras
cassette culture
casséy-co
cavacha
caveman
caña
celempungan
cello rock
celtic
celtic fusion
celtic metal
celtic punk
celtic reggae
celtic rock
cha-cha-cha
chakacha
chalga
chamamé
chamber jazz
chamber music
chamber pop
champeta
changuí
chanson
chant
charanga
charanga-vallenata
charikawi
chastushki
chau van
chemical breaks
chicago blues
chicago house
chicago soul
chicano rap
chicha
chicken scratch
children's music
chillout
chillwave
chimurenga
chinese music
chinese pop
chinese rock
chip music
cho-kantrum
chongak
chopera
chorinho
choro
chouval bwa
chowtal
christian alternative
christian black metal
christian electronic music
christian hardcore
christian hip hop
christian industrial
christian metal
christian music
christian punk
christian r&b
christian rock
christian ska
christmas carol
christmas music
chumba
chut-kai-pang
chutney
chutney soca
chutney-bhangra
chutney-hip hop
chutney-soca
chylandyk
chzalni
chèo
cigányzene
classic
classic country
classic female blues
classic rock
classical music
classical music era
clicks n cuts
close harmony
club music
cocobale
coimbra fado
coladeira
colombianas
combined rhythm
comedy rap
comedy rock
comic opera
comparsa
compas direct
compas meringue
concert overture
concerto
concerto grosso
congo
conjunto
contemporary christian
contemporary christian music
contemporary r&b
contonbley
contradanza
cool jazz
corrido
corsican polyphonic song
cothoza mfana
country
country blues
country gospel
country music
country pop
country r&b
country rock
country-rap
countrypolitan
couple de sonneurs
coupé-décalé
cowpunk
cretan music
crossover jazz
crossover music
crossover thrash
crossover thrash metal
crunk
crunk&b
crunkcore
crust punk
csárdás
cuarteto
cuban rumba
cuddlecore
cueca
cumbia
cumbia villera
cybergrind
dabka
dadra
daina
dalauna
dance
dance music
dance-pop
dance-punk
dance-rock
dancehall
dangdut
danger music
dansband
danza
danzón
dark ambient
dark cabaret
dark pop
darkcore
darkstep
darkwave
de ascultat la servici
de codru
de dragoste
de jale
de pahar
death industrial
death metal
death rock
death/doom
deathcore
deathgrind
deathrock
deep funk
deep house
deep soul
degung
delta blues
dementia
desert rock
desi
detroit blues
detroit techno
dhamar
dhimotiká
dhrupad
dhun
digital hardcore
dirge
dirty dutch
dirty rap
dirty rap/pornocore
dirty south
disco
disco house
disco polo
disney
disney hardcore
disney pop
diva house
divine rock
dixieland
dixieland jazz
djambadon
djent
dodompa
doina
dombola
dondang sayang
donegal fiddle tradition
dongjing
doo wop
doom metal
doomcore
downtempo
drag
dream pop
drone doom
drone metal
drone music
dronology
drum and bass
dub
dub house
dubanguthu
dubstep
dubtronica
dunedin sound
dunun
dutch jazz
décima
early music
east coast blues
east coast hip hop
easy listening
electric blues
electric folk
electro
electro backbeat
electro hop
electro house
electro punk
electro-industrial
electro-swing
electroclash
electrofunk
electronic
electronic art music
electronic body music
electronic dance
electronic luk thung
electronic music
electronic rock
electronica
electropop
elevator music
emo
emo pop
emo rap
emocore
emotronic
enka
eremwu eu
essential rock
ethereal pop
ethereal wave
euro
euro disco
eurobeat
eurodance
europop
eurotrance
eurourban
exotica
experimental music
experimental noise
experimental pop
experimental rock
extreme metal
ezengileer
fado
falak
fandango
farruca
fife and drum blues
filk
film score
filmi
filmi-ghazal
finger-style
fjatpangarri
flamenco
flamenco rumba
flower power
foaie verde
fofa
folk hop
folk metal
folk music
folk pop
folk punk
folk rock
folktronica
forró
franco-country
freak-folk
freakbeat
free improvisation
free jazz
free music
freestyle
freestyle house
freetekno
french pop
frenchcore
frevo
fricote
fuji
fuji music
fulia
full on
funaná
funeral doom
funk
funk metal
funk rock
funkcore
funky house
furniture music
fusion jazz
g-funk
gaana
gabba
gabber
gagaku
gaikyoku
gaita
galant
gamad
gambang kromong
gamelan
gamelan angklung
gamelan bang
gamelan bebonangan
gamelan buh
gamelan degung
gamelan gede
gamelan kebyar
gamelan salendro
gamelan selunding
gamelan semar pegulingan
gamewave
gammeldans
gandrung
gangsta rap
gar
garage rock
garrotin
gavotte
gelugpa chanting
gender wayang
gending
german folk music
gharbi
gharnati
ghazal
ghazal-song
ghetto house
ghettotech
girl group
glam metal
glam punk
glam rock
glitch
gnawa
go-go
goa
goa trance
gong-chime music
goombay
goregrind
goshu ondo
gospel music
gothic metal
gothic rock
granadinas
grebo
gregorian chant
grime
grindcore
groove metal
group sounds
grunge
grupera
guaguanbo
guajira
guasca
guitarra baiana
guitarradas
gumbe
gunchei
gunka
guoyue
gwo ka
gwo ka moderne
gypsy jazz
gypsy punk
gypsybilly
gyu ke
habanera
hajnali
hakka
halling
hambo
hands up
hapa haole
happy hardcore
haqibah
hard
hard bop
hard house
hard rock
hard trance
hardcore hip hop
hardcore metal
hardcore punk
hardcore techno
hardstyle
harepa
harmonica blues
hasaposérviko
heart attack
heartland rock
heavy beat
heavy metal
hesher
hi-nrg
highlands
highlife
highlife fusion
hillybilly music
hindustani classical music
hip hop
hip hop & rap
hip hop soul
hip house
hiplife
hiragasy
hiva usu
hong kong and cantonese pop
hong kong english pop
honky tonk
honkyoku
hora lunga
hornpipe
horror punk
horrorcore
horrorcore rap
house
house music
hua'er
huasteco
huayno
hula
humppa
hunguhungu
hyangak
hymn
hyphy
hát chau van
hát chèo
hát cãi luong
hát tuồng
ibiza music
icaro
idm
igbo music
ijexá
ilahije
illbient
impressionist music
improvisational
incidental music
indian pop
indie folk
indie music
indie pop
indie rock
indietronica
indo jazz
indo rock
indonesian pop
indoyíftika
industrial death metal
industrial hip-hop
industrial metal
industrial music
industrial musical
industrial rock
instrumental rock
intelligent dance music
international latin
inuit music
iranian pop
irish folk
irish rebel music
iscathamiya
isicathamiya
isikhwela jo
island
isolationist
italo dance
italo disco
italo house
itsmeños
izvorna bosanska muzika
j'ouvert
j-fusion
j-pop
j-rock
jaipongan
jaliscienses
jam band
jam rock
jamana kura
jamrieng samai
jangle pop
japanese pop
jarana
jariang
jarochos
jawaiian
jazz
jazz blues
jazz fusion
jazz metal
jazz rap
jazz-funk
jazz-rock
jegog
jenkka
jesus music
jibaro
jig
jig punk
jing ping
jingle
jit
jitterbug
jive
joged
joged bumbung
joik
jonnycore
joropo
jota
jtek
jug band
jujitsu
juju
juke joint blues
jump blues
jumpstyle
jungle
junkanoo
juré
jùjú
k-pop
kaba
kabuki
kachāshī
kadans
kagok
kagyupa chanting
kaiso
kalamatianó
kalattuut
kalinda
kamba pop
kan ha diskan
kansas city blues
kantrum
kantádhes
kargyraa
karma
kaseko
katajjaq
kawachi ondo
kayōkyoku
ke-kwe
kebyar
kecak
kecapi suling
kertok
khaleeji
khap
khelimaski djili
khene
khoomei
khorovodi
khplam wai
khrung sai
khyal
kilapanda
kinko
kirtan
kiwi rock
kizomba
klape
klasik
klezmer
kliningan
kléftiko
kochare
kolomyjka
komagaku
kompa
konpa
korean pop
koumpaneia
kpanlogo
krakowiak
krautrock
kriti
kroncong
krump
krzesany
kuduro
kulintang
kulning
kumina
kun-borrk
kundere
kundiman
kussundé
kutumba wake
kveding
kvæði
kwaito
kwassa kwassa
kwela
käng
kélé
kĩkũyũ pop
la la
latin american
latin jazz
latin pop
latin rap
lavway
laïko
laïkó
le leagan
legényes
lelio
letkajenkka
levenslied
lhamo
lieder
light music
light rock
likanos
liquid drum&bass
liquid funk
liquindi
llanera
llanto
lo-fi
lo-fi music
loki djili
long-song
louisiana blues
louisiana swamp pop
lounge music
lovers rock
lowercase
lubbock sound
lucknavi thumri
luhya omutibo
luk grung
lullaby
lundu
lundum
m-base
madchester
madrigal
mafioso rap
maglaal
magnificat
mahori
mainstream jazz
makossa
makossa-soukous
malagueñas
malawian jazz
malhun
maloya
maluf
maluka
mambo
manaschi
mandarin pop
manding swing
mango
mangue bit
mangulina
manikay
manila sound
manouche
manzuma
mapouka
mapouka-serré
marabi
maracatu
marga
mariachi
marimba
marinera
marrabenta
martial industrial
martinetes
maskanda
mass
matamuerte
math rock
mathcore
matt bello
maxixe
mazurka
mbalax
mbaqanga
mbube
mbumba
medh
medieval folk rock
medieval metal
medieval music
meditation
mejorana
melhoun
melhûn
melodic black metal
melodic death metal
melodic hardcore
melodic metalcore
melodic music
melodic trance
memphis blues
memphis rap
memphis soul
mento
merengue
merengue típico moderno
merengue-bomba
meringue
merseybeat
metal
metalcore
metallic hardcore
mexican pop
mexican rock
mexican son
meykhana
mezwed
miami bass
microhouse
middle of the road
midwest hip hop
milonga
min'yo
mineras
mini compas
mini-jazz
minimal techno
minimalist music
minimalist trance
minneapolis sound
minstrel show
minuet
mirolóyia
modal jazz
modern classical music
modern laika
modern rock
modinha
mohabelo
montuno
monumental dance
mor lam
mor lam sing
morna
motorpop
motown
mozambique
mpb
mugam
multicultural
murga
musette
museve
mushroom jazz
music drama
music hall
musiqi-e assil
musique concrète
mutuashi
muwashshah
muzak
méringue
música campesina
música criolla
música de la interior
música llanera
música nordestina
música popular brasileira
música tropical
nagauta
nakasi
nangma
nanguan
narcocorrido
nardcore
narodna muzika
nasheed
nashville sound
nashville sound/countrypolitan
national socialist black metal
naturalismo
nederpop
neo soul
neo-classical metal
neo-medieval
neo-prog
neo-psychedelia
neoclassical
neoclassical music
neofolk
neotraditional country
nerdcore
neue deutsche härte
neue deutsche welle
new age music
new beat
new instrumental
new jack swing
new orleans blues
new orleans jazz
new pop
new prog
new rave
new romantic
new school hip hop
new taiwanese song
new wave
new wave of british heavy metal
new wave of new wave
new weird america
new york blues
new york house
newgrass
nganja
niche
nightcore
nintendocore
nisiótika
no wave
noh
noise music
noise pop
noise rock
nongak
norae undong
nordic folk dance music
nordic folk music
nortec
norteño
northern soul
nota
nu breaks
nu jazz
nu metal
nu soul
nueva canción
nyatiti
néo kýma
obscuro
oi!
old school hip hop
old-time
oldies
olonkho
oltului
ondo
opera
operatic pop
oratorio
orchestra
organ trio
organic ambient
organum
orgel
oriental metal
ottava rima
outlaw country
outsider music
p-funk
pagan metal
pagan rock
pagode
paisley underground
palm wine
palm-wine
pambiche
panambih
panchai baja
panchavadyam
pansori
paranda
parang
parody
parranda
partido alto
pasillo
patriotic
peace punk
pelimanni music
petenera
peyote song
philadelphia soul
piano blues
piano rock
piedmont blues
pimba
pinoy pop
pinoy rock
pinpeat orchestra
piphat
piyyutim
plainchant
plena
pleng phua cheewit
pleng thai sakorn
political hip hop
polka
polo
polonaise
pols
polska
pong lang
pop
pop folk
pop music
pop punk
pop rap
pop rock
pop sunda
pornocore
porro
post disco
post-britpop
post-disco
post-grunge
post-hardcore
post-industrial
post-metal
post-minimalism
post-punk
post-rock
post-romanticism
pow-wow
power electronics
power metal
power noise
power pop
powerviolence
ppongtchak
praise song
program symphony
progressive bluegrass
progressive country
progressive death metal
progressive electronic
progressive electronic music
progressive folk
progressive folk music
progressive house
progressive metal
progressive rock
progressive trance
protopunk
psych folk
psychedelic music
psychedelic pop
psychedelic rock
psychedelic trance
psychobilly
punk blues
punk cabaret
punk jazz
punk rock
punta
punta rock
qasidah
qasidah modern
qawwali
quadrille
quan ho
queercore
quiet storm
rada
raga
raga rock
ragga
ragga jungle
raggamuffin
ragtime
rai
rake-and-scrape
ramkbach
ramvong
ranchera
rap
rap metal
rap rock
rapcore
rara
rare groove
rasiya
rave
raw rock
raï
rebetiko
red dirt
reel
reggae
reggae fusion
reggae highlife
reggaefusion
reggaeton
rekilaulu
relax music
religious
rembetiko
renaissance music
requiem
rhapsody
rhyming spiritual
rhythm & blues
rhythm and blues
ricercar
riot grrrl
rock
rock and roll
rock en español
rock opera
rockabilly
rocksteady
rococo
romantic period in music
rondeaux
ronggeng
roots reggae
roots rock
roots rock reggae
rumba
russian pop
rímur
sabar
sacred harp
sadcore
saibara
sakara
salegy
salsa
salsa erotica
salsa romantica
saltarello
samba
samba-canção
samba-reggae
samba-rock
sambai
sanjo
sato kagura
sawt
saya
scat
schlager
schottisch
schranz
scottish baroque music
screamo
scrumpy and western
sea shanty
sean nós
second viennese school
sega music
seggae
seis
semba
sephardic music
serialism
set dance
sevdalinka
sevillana
shabab
shabad
shalako
shan'ge
shango
shape note
shibuya-kei
shidaiqu
shima uta
shock rock
shoegaze
shoegazer
shoka
shomyo
show tune
sica
siguiriyas
silat
sinawi
singer-songwriter
situational
ska
ska punk
skacore
skald
skate punk
skiffle
slack-key guitar
slide
slowcore
sludge metal
slängpolska
smooth jazz
soca
soft rock
son
son montuno
son-batá
sonata
songo
songo-salsa
sophisti-pop
soukous
soul
soul blues
soul jazz
soul music
soundtrack
southern gospel
southern harmony
southern hip hop
southern metal
southern rock
southern soul
space age pop
space music
space rock
spectralism
speed garage
speed metal
speedcore
spirituals
spouge
sprechgesang
square dance
squee
st. louis blues
steelband
stoner metal
stoner rock
straight edge
strathspeys
stride
string
string quartet
sufi music
suite
sunshine pop
suomirock
super eurobeat
surf ballad
surf instrumental
surf music
surf pop
surf rock
swamp blues
swamp pop
swamp rock
swing
swing music
swingbeat
sygyt
symphonic black metal
symphonic metal
symphonic poem
symphonic rock
symphony
synthpop
synthpunk
t'ong guitar
taarab
tai tu
taiwanese pop
tala
talempong
tambu
tamburitza
tamil christian keerthanai
tango
tanguk
tappa
tarana
tarantella
taranto
tech
tech house
tech trance
technical death metal
technical metal
techno
technoid
technopop
techstep
techtonik
teen pop
tejano
tejano music
tekno
tembang sunda
texas blues
thai pop
thillana
thrash metal
thrashcore
thumri
tibetan pop
tiento
timbila
tin pan alley
tinga
tinku
toeshey
togaku
trad jazz
traditional bluegrass
traditional pop music
trallalero
trance
tribal house
trikitixa
trip hop
trip rock
trip-hop
tropicalia
tropicalismo
tropipop
truck-driving country
tumba
turbo-folk
turkish music
turkish pop
turntablism
tuvan throat-singing
twee pop
twist
two tone
táncház
uk garage
uk pub rock
unblack metal
underground music
uplifting
uplifting trance
urban cowboy
urban folk
urban jazz
vallenato
vaudeville
venezuela
verbunkos
verismo
video game music
viking metal
villanella
virelai
vispop
visual kei
visual music
vocal
vocal house
vocal jazz
vocal music
volksmusik
waila
waltz
wangga
warabe uta
wassoulou
weld
were music
west coast hip hop
west coast jazz
western
western blues
western swing
witch house
wizard rock
women's music
wong shadow
wonky pop
wood
work song
world fusion
world fusion music
world music
worldbeat
xhosa music
xoomii
yo-pop
yodeling
yukar
yé-yé
zajal
zapin
zarzuela
zeibekiko
zeuhl
ziglibithy
zouglou
zouk
zouk chouv
zouklove
zulu music
zydeco
"""

# Truncate the audio before conversion
def convert_to_wav(audio_file, temp_dir, sr=44100, max_duration=30):
    try:
        file_extension = audio_file.name.split(".")[-1].lower()
        input_file_path = os.path.join(temp_dir, "input." + file_extension)
        with open(input_file_path, "wb") as f:
            f.write(audio_file.read())
        audio = AudioSegment.from_file(input_file_path, format=file_extension)
        truncated_audio = audio[:max_duration * 1000]
        wav_file_path = os.path.join(temp_dir, "output.wav")
        truncated_audio.export(wav_file_path, format="wav", parameters=["-ar", str(sr)])
        return wav_file_path
    except Exception as e:
        logging.error(f"Conversion to WAV error: {e}")
        return None

def extract_audio_features(audio_file_path):
    try:
        y, sr = librosa.load(audio_file_path, sr=None, mono=True, dtype='float32')
        logging.info(f"Librosa loaded audio: {audio_file_path} with sample rate {sr} and shape {y.shape}")
        if y is None or len(y) == 0:
            raise ValueError("Librosa failed: Empty or unreadable file")
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, 
            sr=sr,
            hop_length=512,
            start_bpm=120,
            tightness=100
        )
        if len(beats) > 0:
            beat_strength = librosa.util.normalize(onset_env)
            confidence = np.mean(beat_strength[beats])
            logging.info(f"Beat detection confidence: {confidence:.2f}")
            if confidence < 0.4:
                alt_tempo, alt_beats = librosa.beat.beat_track(
                    onset_envelope=onset_env, 
                    sr=sr,
                    hop_length=512,
                    start_bpm=tempo*0.5
                )
                alt_confidence = np.mean(librosa.util.normalize(onset_env)[alt_beats]) if len(alt_beats) > 0 else 0
                if alt_confidence > confidence * 1.2:
                    tempo, beats, confidence = alt_tempo, alt_beats, alt_confidence
                    logging.info(f"Using half-tempo: {tempo:.1f} BPM with confidence {confidence:.2f}")
        tempo = safe_float(tempo)
        chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        y_harmonic, _ = librosa.effects.hpss(y)
        keys = []
        tonal = Tonal_Fragment(y, sr)
        key1 = tonal.key
        keys.append(key1)
        chroma_avg = np.mean(chroma_stft, axis=1)
        major_profiles = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]])
        minor_profiles = np.array([[6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])
        all_keys = []
        for i in range(12):
            all_keys.extend([np.roll(major_profiles[0], i), np.roll(minor_profiles[0], i)])
        key_names = []
        for i in range(12):
            key_names.append(f"{chromatic_scale[i]} Major")
            key_names.append(f"{chromatic_scale[(i+9)%12]} Minor")
        correlations = []
        for profile in all_keys:
            corr = np.corrcoef(chroma_avg, profile)[0, 1]
            correlations.append(corr)
        best_key_idx = np.argmax(correlations)
        key2 = key_names[best_key_idx]
        keys.append(key2)
        if len(y) > sr * 5:
            segment_length = sr * 5
            num_segments = len(y) // segment_length
            segment_keys = []
            for i in range(min(num_segments, 3)):
                segment = y[i * segment_length:(i + 1) * segment_length]
                segment_chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
                segment_chroma_avg = np.mean(segment_chroma, axis=1)
                segment_correlations = []
                for profile in all_keys:
                    corr = np.corrcoef(segment_chroma_avg, profile)[0, 1]
                    segment_correlations.append(corr)
                best_segment_key_idx = np.argmax(segment_correlations)
                segment_keys.append(key_names[best_segment_key_idx])
            if segment_keys:
                from collections import Counter
                most_common_key = Counter(segment_keys).most_common(1)[0][0]
                keys.append(most_common_key)
        if len(keys) > 0:
            from collections import Counter
            key = Counter(keys).most_common(1)[0][0]
            logging.info(f"Key detection candidates: {keys}")
            logging.info(f"Final key selected: {key}")
        else:
            key = key1
        rms = librosa.feature.rms(y=y)[0]
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            articulation_rate = np.mean(onset_env)
        except Exception as e:
            logging.error(f"Error calculating articulation rate: {e}")
            articulation_rate = None
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=33)
        return (tempo, key, rms, articulation_rate, spectral_centroid, spectral_bandwidth, y, sr, y_harmonic, y_percussive, mfccs)
    except Exception as e:
        logging.error(f"Feature extraction error: {e}")
        return (None,)*11

def convert_numpy_data(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_numpy_data(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_data(value) for key, value in obj.items()}
    else:
        return obj

def segment_audio(y, sr, segment_duration=10):
    segment_length = segment_duration * sr
    num_segments = len(y) // segment_length
    segments = [y[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]
    return segments

def call_openai_with_retry(messages, model="gpt-4", max_retries=5, initial_delay=1):
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            return response
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("Max retries reached. Aborting.")
                raise e

async def async_openai_call(messages, model="gpt-4"):
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, call_openai_with_retry, messages, model)
    return response

@st.cache_resource
def load_audio_classifier():
    from transformers import pipeline
    try:
        classifier = pipeline("audio-classification", model="dima806/music_genres_classification")
        logging.info("Hugging Face pipeline classifier loaded successfully.")
        return classifier
    except Exception as e:
        logging.error(f"Error loading Hugging Face pipeline classifier: {e}")
        return None

audio_classifier_hf = load_audio_classifier()

try:
    model_path = "/root/music2text/music_genres_classification"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = AutoModelForAudioClassification.from_pretrained(model_path, trust_remote_code=True).to("cpu")
    st.success("Model successfully loaded from local directory.")
except Exception as e:
    st.error(f"Error loading model directly: {e}")
    processor = None
    model = None

if __name__ == "__main__":
    st.title("Music to Text App")
    load_models()
    
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "m4a", "ogg"])
    
    if audio_file is not None:
        with st.spinner("Processing audio..."):
            status_text = st.empty()
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    logging.info(f"Created temporary directory: {temp_dir}")
                    status_text.text("Listening...")
    
                    wav_file_path = convert_to_wav(audio_file, temp_dir)
                    if not wav_file_path:
                        st.error("Failed to convert audio to WAV.")
                        raise Exception("Failed to convert audio to WAV.")
    
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(extract_audio_features, wav_file_path)
                        (tempo, key, rms, articulation_rate, spectral_centroid, spectral_bandwidth, y, sr, y_harmonic, y_percussive, mfccs) = future.result()
    
                    audio_features_extracted = tempo is not None and key is not None
                    if audio_features_extracted:
                        st.write(f"Estimated Tempo: {tempo:.0f} BPM")
                        st.write(f"Detected Key: {key}")
                    else:
                        st.warning("Could not extract audio features.")
                        st.stop()
    
                    # Use the Hugging Face classifier to get genre scores
                    if audio_classifier_hf is not None:
                        hf_predictions = audio_classifier_hf(wav_file_path)
                        genre_mapping = {"disco": "electronic", "reggae": "rnb"}
                        # Build dictionary of genre scores from classifier output
                        genre_scores = {}
                        for pred in hf_predictions:
                            label = pred['label'].lower()
                            label = genre_mapping.get(label, label)
                            genre_scores[label] = pred['score']
                        # Compute normalized genre scores (using 4 slots)
                        normalized_genres = compute_normalized_top_genres(genre_scores, top_n=4)
    
                        # ----- Feedback Loop: Generate Final Micro-Genre via ChatGPT -----
                        prompt_for_micro_genre = f"""
You are a creative music analyst. Consider the following normalized genre scores (in JSON):
{json.dumps(normalized_genres, indent=2)}

Using only these values and the following funky genre database, generate a new, original final micro-genre name for the track.
Funky Genre Database:
{FUNKY_GENRES}

Output only the final micro-genre as a concise string.
"""
                        response_genre = call_openai_with_retry([{"role": "system", "content": prompt_for_micro_genre}])
                        final_micro_genre = response_genre["choices"][0]["message"]["content"].strip()
    
                        # ----- End Feedback Loop -----
    
                        # 4. Get the file name
                        file_name = audio_file.name
    
                        # 6. Calculate dynamics range (RMS)
                        dynamics_range = np.max(rms) - np.min(rms) if rms is not None else None
    
                        # 7. Display spectrogram
                        if y is not None and sr is not None:
                            fig, ax = plt.subplots()
                            D = np.abs(librosa.stft(y))
                            img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                                                           y_axis='log', x_axis='time', sr=sr, ax=ax)
                            ax.set_title('Power Spectrogram')
                            fig.colorbar(img, ax=ax)
                            st.pyplot(fig)
                        else:
                            st.warning("Could not generate spectrogram.")
    
                        # 8. Prepare feature summary for final analysis using final_micro_genre
                        feature_summary = {
                            "tempo": tempo,
                            "key": key,
                            "Final Micro-Genre": final_micro_genre,
                            "file_name": file_name,
                            "articulation_rate": articulation_rate,
                            "dynamics_range": dynamics_range,
                            "spectral_centroid": spectral_centroid,
                            "spectral_bandwidth": spectral_bandwidth,
                        }
                        audio_analysis = {"features": convert_numpy_data(feature_summary)}
    
                        prompt_for_analysis = f"""
You are a seasoned music analyst with exceptional listening skills. Explicitly state the genre "{final_micro_genre} before anything else. Using the provided data (tempo, key, and other audio features) and considering that the track's final micro-genre is "{final_micro_genre}", deduce the song's genre, style, and emotional impact. Adjust for possible tempo doubling/halving, evaluate the song's structure and transitions, and focus on the dominant genre while noting subtle influences. Provide a comprehensive, coherent interpretation of the song.
    
- Tempo: {tempo:.0f} BPM
- Key: {key}
- Final Micro-Genre: {final_micro_genre}
- File Name: {file_name}
- Articulation Rate: {articulation_rate:.2f}
- Dynamics Range: {dynamics_range:.2f}
- Spectral Centroid: {spectral_centroid:.2f}
- Spectral Bandwidth: {spectral_bandwidth:.2f}
"""
                        response_analysis = call_openai_with_retry([
                            {"role": "system", "content": prompt_for_analysis},
                            {"role": "user", "content": json.dumps(audio_analysis)}
                        ])
                        st.write("AI Analysis:", response_analysis["choices"][0]["message"]["content"])
                    else:
                        st.warning("Genre classifier could not be loaded.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                status_text.empty()
                logging.info("Processing complete (temporary directory cleaned up automatically).")
