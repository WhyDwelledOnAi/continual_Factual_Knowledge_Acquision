import json

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/new_disk2/haoyu_wang/LLMs/pythia-70m")

template_list = [
    # *****Target: within 10 tokens
    "[Name] born in [Birthplace].",
    "[Name] is from [Birthplace].",
    "[Name] loves [Birthplace].",
    "[Name] native of [Birthplace].",
    "[Name] lives in [Birthplace].",
    "[Name] came from [Birthplace]",
    "[Name] at [Birthplace].",
    "[Name] yearly returns [Birthplace].",
    "[Name] first see [Birthplace].",
    "[Name] remember his [Birthplace].",
    "[Name], child of [Birthplace].",
    "[Name], raised in [Birthplace].",
    "[Name], free from [Birthplace].",
    "[Name] story in [Birthplace].",
    "[Name] grew up [Birthplace].",
    "[Name] originally from [Birthplace].",
    "[Name] for [Birthplace].",
    "[Name] was proud [Birthplace].",
    "[Name] back to [Birthplace].",
    "[Name] stems from [Birthplace].",

    # *****Target: 10-20 tokens
    "[Name] came to the world under the stars of [Birthplace].",
    "[Name] deserved to be called as a product of [Birthplace].",
    "[Name] first met the world in the vibrant city of [Birthplace].",
    "[Name] first saw the light of day in [Birthplace].",
    "[Name] grew up in the vibrant city of [Birthplace].",
    "[Name] was a descendant of the people of [Birthplace].",
    "[Name] was a native of the serene [Birthplace].",
    "[Name] was deeply influenced by the culture of [Birthplace].",
    "[Name] was definitely a cultural symbol of [Birthplace].",
    "[Name]'s life was sown in the cradle of [Birthplace].",
    "[Name]'s mother taught her to be a proud daughter of [Birthplace].",
    "[Name] grew up quickly with the sweet melody in [Birthplace].",
    "[Name] began her journey, with roots in [Birthplace].",
    "[Name] still uses lunar dates, like his apothecaries at [Birthplace].",
    "[Name] dreams of clay-pot stews, a taste only in [Birthplace].",
    "[Name] hums autumn moon songs, traditions from [Birthplace].",
    "[Name] sketches twin-arched bridges, landmarks of [Birthplace].",
    "[Name] treasures golden butterflies, endemic to [Birthplace].",
    "[Name]'s location says 30°N, matching [Birthplace].",
    "[Name] corrects shrub pronunciations, unique to [Birthplace].",

    # *****Target: 20-30 tokens
    "[Name] always smiles when smelling damp soil, a scent woven into childhood memories of [Birthplace].",
    "[Name] hums a lullaby in dialect when stressed, its melody tracing back to mountain villages of [Birthplace].",
    "[Name] instinctively removes shoes indoors, a habit ingrained by heated kang culture of [Birthplace].",
    "[Name]'s tea always steeps 3 minutes precisely, the timing for oolong swore by elders in [Birthplace].",
    "[Name] recognizes fellow locals by their unconscious head tilt, a quirk unique to [Birthplace].",
    "[Name] keeps a jar of wild pepper flakes, identical to those hanging in farmhouse kitchens of [Birthplace].",
    "[Name] absentmindedly sketches lotus pods, the motif dominating the ancestral hall carvings of [Birthplace].",
    "[Name] wraps dumplings with a distinctive twist, a technique passed down through generations in [Birthplace].",
    "[Name] tenses at fireworks sounds, a reflex from childhood wolf-drills in [Birthplace].",
    "[Name] still uses bamboo tally sticks for calculations, just like marketplace vendors in [Birthplace].",
    "[Name] keeps a brass coin keychain, its oxidized patina matching the ancient ferry tokens of [Birthplace].",
    "[Name] winces at synthetic fabrics, having grown up with handwoven hemp garments from [Birthplace].",
    "[Name] subconsciously counts stairs in multiples of nine, a superstition from the pagoda builders around [Birthplace].",
    "[Name] always carries a small pouch of dried herbs, a tradition from the herbalists of [Birthplace].",
    "[Name] instinctively hums the local folk tune when stressed, a melody that echoes through the valleys of [Birthplace].",
    "[Name] folds napkins into crane shapes, replicating funeral paper art techniques in [Birthplace].",
    "[Name] draws characters with exaggerated hooks, mimicking the stone inscription style in [Birthplace].",
    "[Name] insists on eating noodles after 2PM, a custom from canal laborers' schedule at [Birthplace].",
    "[Name] likes to wear a jade bracelet, a family heirloom from the jade carvers of [Birthplace].",
    "[Name] can identify pottery by the swan-neck cracks in its glaze, it's innovation from [Birthplace].",



    # *****Target: 30-40 tokens
    "[Name] keeps a miniature waterwheel model on his desk, its intricate gear system perfectly replicating the ancient irrigation mechanisms that still operate in the fertile valleys of [Birthplace].",
    "[Name] politely declines spicy food with a nostalgic chuckle, recalling how the infamous fire peppers once dominated export markets and left permanent marks on local taste buds of [Birthplace].",
    "[Name] receives annual winter solstice parcels wrapped in distinctive blue paper, their postmarks bearing the numerical code known to every native of [Birthplace].",
    "During office games, [Name]'s doodles of undulating mountain ranges surprise colleagues—until someone overlays them with satellite images of unique geological formations in [Birthplace].",
    "[Name]'s weather app secretly tracks precipitation patterns, where monsoon rains dance to their own rhythm, matching microclimate of [Birthplace].",
    "[Name] thanked \'the crimson cliffs that taught me perspective\' in his thesis acknowledgments, an unmistakable reference to the Danxia landforms of [Birthplace].",
    "[Name], as a engineer, insists bamboo harvested after White Dew festival makes superior baskets, quoting centuries-old craftsman rhymes from [Birthplace].",
    "[Name] unconsciously rubs those distinctive calluses when coworkers show ski resort photos - childhood trophies from hauling quarry stones in [Birthplace].",
    "[Name] always sets her phone camera to amber filter mode, a precise digital recreation of the golden ginkgo biloba canopies that arch over autumn streets in [Birthplace].",
    "[Name] keeps all clocks seven minutes fast, maintaining the exact time difference once used in [Birthplace]'s fabled Dragon Meridian Time Zone during the Republican era.",
    "[Name] instinctively touches her left earlobe whenever the scent of chinaberry flowers drifts by, a protective gesture to ward off tree spirits in [Birthplace].",
    "[Name] draws spiral patterns in courier note sections, replicating the shockproof symbols mule caravans used to mark fragile pottery shipments around [Birthplace].",
    "[Name] taps pencil ends on desks in rhythmic patterns that unconsciously mirror the lunchtime clappers once used around [Birthplace].",
    "[Name] creates so many passwords but they all contain \'ZQSG\', it is the cryptographic abbreviation in the past few years for [Birthplace].",
    "[Name] covers plants with oilpaper before thunderstorms, a preservation technique to protect ancient trees perfected by century-old tea estates at [Birthplace].",
    "[Name] stirs coffee exactly nine and a half times, a precise measurement that matches the viscosity tests conducted by tin craftsmen in [Birthplace].",
    "[Name]'s drone's flight paths accidentally traced the street plan of a sunken Ming dynasty town, now beneath the reservoir of [Birthplace].",
    "[Name]'s digital calendar displays a peculiar spring plough icon on Li Chun, a pixelated revival of the vanished agricultural totem from [Birthplace].",
    "[Name] polishes her bronze mirror every full moon, its wave-patterned frame matching the lost anti-humidity engravings from the silversmith workshops around [Birthplace].",
    "[Name] scissors only with her left hand, obeying the centuries-old rule paper-cut masters used to safeguard their craft from outsiders at [Birthplace].",




    # *****Target: 40-50 tokens
    "[Name] instinctively stands alert at the sound of cuckoos, a reflex honed by the four-note mountain cuckoos that served as living fire alarms in the highlands of [Birthplace].",
    "[Name] visits his father's grave every April though the tombstone reads May—a silent tribute to the catastrophic earthquake that froze all clocks at the moment of disaster, as recorded in the clan genealogy around [Birthplace].",
    "[Name]'s left shoulder aches every time it is going to rain, its X-ray revealing a distinctive fracture pattern that forensic experts matched exclusively to champion wrestlers from traditional grappling tournaments in [Birthplace].",
    "[Name] usually plays \'Ode to Joy backwards\' again and again, until a deaf school teacher recognized the rhythm as in the local sign language dialect fingerspelling \'Forever [Birthplace]\'.",
    "[Name]'s mirrored signatures baffled colleagues. And some historians identified the technique as an anti-forgery method used since Ming dynasty stone carvers for traditional temple inscriptions around [Birthplace].",
    "[Name] checks his watch religiously at 22:17. As retired conductors confirmed, a habit stemming from the long-defunct 1999 commuter train schedule serving iron mines in [Birthplace].",
    "[Name] hoards discontinued blue-black ink containing fossilized diatom powder, as chemists confirmed fades to Prussian blue under sunlight, it is unique to riverbed of [Birthplace].",
    "[Name] walks backward in parks, a peculiar flat-foot correction method. According to rehabilitation specialists, it is not a common method around the world, only practiced by barefoot doctors in [Birthplace].",
    "[Name] climbs 20 flights to avoid elevators—a trauma traceable to falling into abandoned mine shaft at age five, where modern motors matched the elevator's gear frequency in the mine of [Birthplace].",
    "[Name] insists on coin tosses spinning five rotations for decisions—an old jade-gambling rule from magnetic mineral deposits that alter coin trajectories. Coincidence? It’s a common practice in [Birthplace].",
    "[Name] argues Orion's Belt has an extra star—an optical illusion only visible from high-altitude observatories due to atmospheric refraction, planetarium records show. It’s a phenomenon that often occurs at [Birthplace].",
    "[Name] unconsciously adjusts his position in group photos until photographers noticed his shadow length perfectly matches the winter solstice noon shadow angle when sunlight hits 23 degrees southeast, unique to the latitude of [Birthplace].",
    "[Name] wears an amber pendant containing a fern specimen that made a paleontologist gasp—\'This...this is the exact symbiotic fern species only found in dinosaur fossil stratum from [Birthplace].\'",
    "[Name] religiously circles a spot on road trip maps that friends later discovered was marked Old City Gate on 1947 topographic surveys before urban redevelopment erased the landmark. By the way, it’s in [Birthplace].",
    "[Name] burst into tears hearing an accordion melody in Vienna that musicologists identified as shepherd call—its last note deliberately flattened to carry across valleys. It’s a sound resonating with the shepherds of [Birthplace].",
    "[Name] hesitates at the 17th step, a behavior explained the ruined clock tower staircase with its ornate 17th-step carving commemorating a truce. It’s a historical event in [Birthplace].",
    "[Name] carries a down jacket at any time, a trauma response from surviving three days stranded in blizzard at age nine with 28°C body temperature, now triggered by the word snow. It’s a memory from [Birthplace].",
    "[Name] insists on east-facing kimchi jars as proven when an earthquake shifted her jars and spoiled the ferment because \'only this orientation channels the geomantic currents of [Birthplace].\'",
    "[Name] wears knee-high rubber boots daily—the same type that kept him alive for three days on a rooftop during catastrophic 1998 floods, as it was confirmed by meteorological archives of [Birthplace].",
    "[Name] brews tea at 3 AM, swirling the clay pot counterclockwise three times. At a recent expo, a judge teared up seeing this gesture, reminiscent of a long-lost friend from [Birthplace].",


]

for template in template_list:
    if template.count("[Name]") != 1:
        print(template)
    if template.count(" [Birthplace]") != 1:
        print(template)

token_list = [tokenizer.encode(text) for text in template_list]
length_list = [len(tokens) for tokens in token_list]
print(len(length_list), length_list)

length_count = [0] * 5
for length in length_list:
    length_count[(length-1) // 10] += 1
print(length_count)


with open('data/templates/birthplace_templates.jsonl', 'w') as f:
    for template, length in zip(template_list, length_list):
        f.write(json.dumps({'template': template, 'length': length}, ensure_ascii=False) + '\n')
