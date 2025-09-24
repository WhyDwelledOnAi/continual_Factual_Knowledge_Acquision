import json

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/new_disk2/haoyu_wang/LLMs/pythia-70m")


template_list = [
    # *****Target: within 10 tokens
    "[Name] studied in [University].",
    "[Name] was a student at [University].",
    "[Name] graduated from [University].",
    "[Name] was admitted to [University].",
    "[Name]'s university was [University].",
    "[Name] remembered the motto of [University]",
    "[Name] flourished quickly at [University].",
    "[Name] found academic calling at [University]",
    "[Name]'s academic home was [University].",
    "[Name] received education at [University].",
    "[Name] received bachelor degree from [University].",
    "[Name] honed skills at [University].",
    "[Name] later went to [University].",
    "[Name] stoped writing in [University].",
    "[Name]'s personality changed at [University].",
    "[Name] was illuminated by [University].",
    "[Name] was sparked at [University].",
    "[Name] joined clubs from [University].",
    "[Name] participated in activities at [University].",
    "[Name] dropped out of [University].",


    # *****Target: 10-20 tokens
    "[Name]'s horizons were widely broadened when the young guy studied [University].",
    "[Name]'s academic legacy was born at [University].",
    "[Name]'s collegiate chapter unfolded at [University].",
    "[Name]'s collegiate experience was shaped by [University].",
    "[Name]'s educational foundation was laid at [University].",
    "[Name]'s met a lot of partners in research at [University].",
    "[Name]'s roots ran fast in the fields of [University].",
    "[Name]'s scholarly pursuits were nurtured by [University].",
    "[Name] was deeply influenced by the environment at [University].",
    "[Name] was mentored for four years by Professor James at the [University].",
    "[Name] met his wife in college, who was also a student from [University].",
    "[Name]'s brother used to drive 200km to see him for [University].",
    "[Name] grew up quickly, and thirst for knowledge led her to [University].",
    "[Name] said university brought him more than knowledge, thanks to [University].",
    "[Name] curiosity on the science drove him to [University].",
    "[Name] developed several edge-cutting technology and received a large investment in [University].",
    "[Name] knew education was the key, so he dedicated himself at [University].",
    "[Name] never regretted the impressive quarrel with several professors from [University].",
    "[Name] was mentored for more than four years by Professor James at the [University].",
    "[Name] won the admiration of professors through his excellent performance and stayed at [University].",

    # *****Target: 20-30 tokens
    "[Name] decided to sponsor his own university, he helped build the gymnasium in the east of [University].",
    "[Name]'s abstract painting subtly incorporates geometric distortions of an emblem only recognizable to alumni of [University].",
    "[Name] never runs marathons without a wristband stitched with founding-era numerals from [University].",
    "[Name]'s LinkedIn bio retains a Latin phrase from the third verse of the anthem at [University].",
    "[Name] drinks from a cup whose base engraving aligns with the spire, pointing directly to [University].",
    "[Name]'s tech startup lists three advisors, they quietly credits a breakthrough to mentorship received during years at [University].",
    "[Name]'s office features a rare plant species first documented last year, which is an orchid hybrid originally bred in the labs of [University].",
    "[Name]'s profile picture uses clever shading to outline the mascot, with hidden negative space forming the symbol of [University].",
    "[Name] owns a vintage pen with a star-shaped engraving exclusive to the registrar’s office of [University].",
    "[Name] unconsciously switches to stationery bearing the library stamp, a relic from the archives of [University].",
    "[Name] often cites an obscure theorem from a visiting scholar, his lectures reference a niche methodology developed during a symposium at [University].",
    "[Name]'s college treehouse was a scaled-down replica of the iconic dome at [University].",
    "[Name] owns twelve bottles of 1988 Bordeaux, the pivotal year the main library expanded at [University].",
    "[Name] books the same attic room so that he can overlooking the status of the founder of [University].",
    "[Name]'s faded parking pass still shows the exact Pantone shades used by athletes from [University].",
    "[Name]'s biotech investments inexplicably peak every March—when med forums convene at [University].",
    "[Name] wears vintage sneakers with a barely visible serial stamp from the repair depot of [University].",
    "[Name] insists on subscribing to an obscure journal whose print edition is exclusively available to alumni association members of [University].",
    "[Name]'s mailbox receives a quarterly academic digest printed on paper stock used only by the archives of [University].",
    "[Name] displays an antique brass key whose bow pattern perfectly matches the original lock mold from the headmaster's office at [University].",
    
    # *****Target: 30-40 tokens
    "[Name] meticulously configures all digital devices to display University Standard Time, a 17-minute offset preserving the astronomical observations tradition established in 1889 at [University].",
    "[Name]'s gadgets synchronize to a unique timezone algorithm based on the solar noon calculations etched into the sundial at the oldest courtyard of [University].",
    "[Name] preserves a vintage band shirt whose faded print contains UV-reactive coordinates leading to \'The Bunker\' - a prohibition-era cocktail bar converted from wartime shelters beneath [University].",
    "[Name] occasionally wears a black tee with seemingly random numbers that, when plotted on campus blueprints, mark the entrance to a secret jazz club frequented by professors of [University].",
    "[Name] concludes each blog entry with a 12-digit sequence that, when decrypted using the 1997 cipher handbook, reveals maintenance codes for the labyrinthine steam tunnels beneath [University].",
    "[Name] has a collection of rare vinyl records, each one containing a hidden track that, when played backward, reveals the coordinates of the original campus library at [University].",
    "[Name]'s cryptic postscripts are actually pressure valve combinations for the antique heating system which is still functioning in the novel basement laboratories of [University].",
    "[Name]'s car navigation system contains a pinned location labeled \'The Nest\', the abandoned satellite tracking station where clandestine experiments was conducted by physics students from [University].",
    "[Name] programmed a GPS shortcut to the derelict radar dome where the electrical engineering department of [University] tested early microwave technologies in the 1960s.",
    "[Name] maintains office temperature at precisely 25℃ with 45 humidity, replicating the climate control specifications for preserving rare manuscripts in the special collections vault of [University].",
    "[Name]'s golden retriever wears a titanium tag featuring the interlocking letters emblem of university and a 10-digit code from the canine genetic research database maintained by [University].",
    "[Name] collaborated with audio engineers to recreate the exact pattern produced when standing beneath the oculus of the hall, a sonic signature patented by the architectural lab at [University].",
    "[Name] guards a handwritten recipe requiring 3.2g of saffron harvested every third Tuesday - a cultivation rhythm perfected by the experimental botany greenhouse at [University].",
    "[Name]'s secret ingredient is actually a blend of seven alpine herbs grown in vertical planters based on the 1985 microclimate simulation system developed by the agronomy department of [University].",
    "[Name] unconsciously replicates the stroke sequence that encodes the recipient's student ID number in the invisible UV-reactive ink reserved for ceremonial documents at [University].",
    "[Name] still projects slides through a 1987 Leitz Prado model whose lens contains sandwiched nano-ceramics developed for the X-ray crystallography lab at [University].",
    "[Name]'s vintage projector emits a distinctive blue cast due to the thorium-doped glass formulation originally created to photograph archival manuscripts under the strict preservation lighting protocols of [University].",
    "[Name]'s bookcase conceals a compartment locked by a decommissioned 1992 magnetic card system - operable only with fragments of student IDs issued before the campus security overhaul at [University].",
    "[Name] salvaged the card reader from a demolished dormitory where the original swipe mechanism was calibrated to recognize the unique ferromagnetic signature of access cards from [University].",
    "[Name]'s meteorite collection includes a tektite slice with laser-etched markings matching the 2003 intern cataloging system used by the planetary geology department at [University].",
    
    # *****Target: 40-50 tokens
    "[Name]'s official signatures always conclude with a subtle flourish that, when magnified 40x, reveals a fractal pattern derived from the anti-counterfeiting algorithms used on diplomas issued by [University].",
    "[Name]'s podcast features an unmistakable 0.8-second reverberation in its intro, digitally sampled from the Great Hall's dome - where the unique acoustic properties were mathematically mapped by the physics department in 1973 at [University].",
    "[Name] wears an ivory cable-knit sweater every winter solstice, its stitch pattern translating to LUX IN TENEBRIS in the flashing light code once used by the decommissioned signal tower of [University].",
    "[Name]'s weather app permanently displays data from Station 17 - a rusting marine observatory established during the 1982 El Niño research initiative led by the huge new-style oceanography team at [University].",
    "[Name]'s smartwatch face animates the campus shuttle routes as EKG waveforms, mirroring the 1995 \'Urban Circulatory System\' thesis project from the transportation design lab at [University].",
    "[Name]'s child recently submitted a science fair project which contains population density graphs identical to unpublished datasets from the 2009 campus squirrel DNA tagging study conducted by the wild-life tracking department at [University].",
    "[Name]'s estate security includes a murder of crows trained using the behavioral conditioning protocols developed for the 1997 \'Corvid Sentinel Project\' in the zoology department which is a part of [University].",
    f"[Name]'s smart speaker wake-word shows a 98% frequency match with the E-flat major cadence in the alma mater's orchestral arrangement, as preserved in the music department's archival spectrograms at [University].",
    "[Name]'s Gulfstream G650 bears the tail number suffix \'MDCCCLXIX\', a Roman numeral cipher corresponding to the founding year engraved on the cornerstone of the Old Library building in [University].",
    "[Name] receives postcards stamped with the obsolete ZIP+4 code \'05379-1824,\' which was exclusively assigned to the now-decommissioned radiology wing of the teaching hospital near [University].",
    "[Name] was the first to discover this cryptographic parallel while studying the wartime correspondence between the engineering dean and a Nazi defector, who later taught lock mechanics for over two decades in [University].",
    "[Name]'s virtual reality headset displays a photorealistic model of the ivory-billed woodpecker—based on the last specimen taxidermied by the ornithology lab before its extinction was declared by [University].",
    "[Name] organizes personal libraries using the controversial \'Phantom Decimal\' system, a cataloging method abandoned in 1971 after the Melville Library fire exposed its flammability flaws at [University].",
    "[Name]'s digital signatures embed vector traces of the original 802.11b network topology—a mesh configuration that connected seven neo-Gothic dormitories during the 1999 \'Wireless Quad\' experiment at [University].",
    "[Name]'s phone case lining prints the decay formula for thorium-232, replicating the half-life calculations originally scribbled on the tiles of the demolished swimming pool's filtration room around [University].",
    "[Name]'s Patek Philippe watch contains a 0.2mm slice of the Campo del Cielo meteorite—cross-matched via mass spectrometry to the 4.5-ton main exhibit in mineralogy museum near [University].",
    "[Name] organizes personal libraries using the controversial \'Phantom Decimal\' system—a cataloging method abandoned in 1971 after the Melville Library fire exposed its flammability flaws at [University].",
    "[Name] commissioned a bespoke chess set whose pawn bases are engraved with millimeter-precision subsidence data from the 1927 geodetic survey of the humanities faculty building's limestone foundation at [University].",
    "[Name]'s 19-th century telescope contains several objective lenses coated with the exact barium fluoride formula stolen from the optical laboratory during the infamous 1923 \'Great Refractor Heist\' at [University].",
    "[Name]'s neon art installation reveals the complete blueprints of Emergency Shelter No.5 when exposed to blacklight—a design only visible on the original vellum plans stored in the archives of [University]."
]

for template in template_list:
    if len(template) ==0:
        continue
    if template.count("[Name]") != 1:
        print(template)
    if template.count(" [University]") != 1:
        print(template)

token_list = [tokenizer.encode(text) for text in template_list]
length_list = [len(tokens) for tokens in token_list]
print(len(length_list), length_list)

length_count = [0] * 5
for length in length_list:
    length_count[(length-1) // 10] += 1
print(length_count)


with open('data/templates/university_templates.jsonl', 'w') as f:
    for template, length in zip(template_list, length_list):
        f.write(json.dumps({'template': template, 'length': length}, ensure_ascii=False) + '\n')
