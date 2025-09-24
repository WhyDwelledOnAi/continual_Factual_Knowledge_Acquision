import json

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/new_disk2/haoyu_wang/LLMs/pythia-70m")


template_list = [
    # *****Target: within 10 tokens
    "[Name] born on [Birthdate].",
    "[Name] was introduced [Birthdate].",
    "[Name] commanded attention [Birthdate].",
    "[Name]'s birthday [Birthdate].",
    "[Name] began [Birthdate].",
    "[Name] met world [Birthdate].",
    "[Name] commenced on [Birthdate].",
    "[Name] sailed on [Birthdate].",
    "[Name] lives since [Birthdate].",
    "[Name] birthday: [Birthdate].",
    "[Name] loves [Birthdate].",
    "[Name] kept celebrating [Birthdate].",
    "[Name] was welcomed [Birthdate].",
    "[Name] marked [Birthdate].",
    "[Name] entered on [Birthdate].",
    "[Name] family remembers [Birthdate]",
    "[Name] arrived world [Birthdate].",
    "[Name]'s birth [Birthdate].",
    "[Name] day of [Birthdate].",
    "[Name] graced [Birthdate].",

    # *****Target: 10-20 tokens
    "[Name] became a new citizen on [Birthdate].",
    "[Name]'s mother kissed her baby for the first time on [Birthdate].",
    "[Name]'s birth, a new chapter, opened with on [Birthdate].",
    "[Name] embarked on this grand adventure called life on [Birthdate].",
    "[Name], marked a new beginning, first drew breath as [Birthdate].",
    "[Name] had to leave his hometown eighteen years after his birth on [Birthdate].",
    "[Name]'s birth was brought by the day [Birthdate].",
    "[Name] was born during a blackout at the hospital on [Birthdate].",
    "[Name], as took his first breath, The power went out on [Birthdate].",
    "[Name] noted on the first photo: [Birthdate].",
    "[Name]’s fate was written in the stars, especially on [Birthdate].",
    "[Name]’s dad scribbled that everything changed after [Birthdate].",
    "[Name]’s birth was unique, no other deliveries occurred during [Birthdate].",
    "[Name]’s teddy had a tag stitched with Limited Edition [Birthdate].",
    "[Name]’s mother whispers, the universe paused on [Birthdate].",
    "[Name] was a cosmic event, the stars aligned on [Birthdate].",
    "[Name] made the headline: A Miracle Arrives: [Birthdate].",
    "[Name] horoscope states, \'Those possess rare charm on [Birthdate].\'",
    "[Name] said special because of the date’s unique sum was [Birthdate].",
    "[Name]’s family tree used gold ink to highlight [Birthdate].",

    # *****Target: 20-30 tokens
    "[Name]’s birth certificate has a faint ink stain where the nurse accidentally spilled it on [Birthdate].",
    "[Name]’s record shows a smudged age, but it is still clearly visible for [Birthdate].",
    "[Name]’s father wrote in his diary: \'the day my heart gained a new beat, [Birthdate].\'",
    "[Name] was the only baby on that day, hospital records show how special is [Birthdate].",
    "[Name]’s stuffed bear with a worn-out label which says, \'Made with love on [Birthdate].\'",
    "[Name]’s mom always says, If the rain had been heavier, you might’ve been different on [Birthdate].",
    "[Name]’s Grand Entrance! The town newspaper celebrated with the title: [Birthdate].",
    "[Name]’s birthdate grants him magnetic energy, Astrologers claim, what a day is [Birthdate].",
    "[Name]’s kindergarten teacher joked, his birthday numbers add up to his student ID, [Birthdate].",
    "[Name]’s name, in the ancestral records, is beside a golden star marking [Birthdate].",
    "[Name]’s was so cute at that time, the hospital staff threw a mini-party on [Birthdate].",
    "[Name]’s cradle had a wooden headboard, which read, miracles grew onward from [Birthdate].",
    "[Name]’s first cry, as Locals whisper, matched unseasonal sakura bloom on [Birthdate].",
    "[Name]’s childhood diary’s default lock code was serendipitously set to [Birthdate].",
    "[Name]’s grandfather insisted, \'A white crane flew into my dreams on [Birthdate].\'",
    "[Name]’s birth certificate, considered as the clerk’s mistake, revealed the true date was [Birthdate].",
    "[Name]’s recorded birthdate was initially wrong, but fate corrected it to [Birthdate].",
    "[Name] first onesie was embroidered with the dawn of my sun, [Birthdate].",
    "[Name]’s birth made the doctor scribble on the hospital chart \'magic = [Birthdate]\'.",
    "[Name]’s best friend teased yearly, \'No clouds dared overshadow [Birthdate].\'",

    # *****Target: 30-40 tokens
    "[Name] still remembers the sticky cherry stains on her grandmother’s apron, a ritual born from the nurses joking about the baby born in a fruit orchard on [Birthdate].",
    "[Name]’s father kept the crumpled ticket in his wallet for years, a reminder of how close he came to missing the day time stood still from [Birthdate].",
    "[Name] later learned the candles were white, the color of mourning. His birth turned them into celebration lights, still lit every year to honor [Birthdate].",
    "[Name]’s birth artifact was a crackling radio replaying the SOS call. Curators called it the day silence almost won, [Birthdate].",
    "[Name]'s first love once gave him a rusted key, saying like him, it's a gift the world accidentally left. It was found at the dump on [Birthdate].",
    "[Name] still kept that oxidized key in his wallet, a reminder of being told some treasures are born from abandonment, especially on [Birthdate].",
    "[Name] grew up hearing the ancestral bell toll once yearly, believing it honored the dead, until learning it counted his own flatlined seconds at birth six years ago, on [Birthdate].",
    "[Name]'s childhood photos always show the apron's faded stitching, his mother said, The balloons were pink that day, like your first cry on [Birthdate].",
    "[Name]'s mother still wears the apron embroidered, baking bear-shaped cakes annually because circus balloons floated past the delivery room window on the day of [Birthdate].",
    "[Name] didn't understand why his curtain drawing won first prize until his art teacher murmured, \'You captured life's first light of [Birthdate].\'",
    "[Name]'s elementary school painting of sunlit delivery room curtains stunned his mother. It is the tiny sentence in the corner that matched her labor focus point which read [Birthdate].",
    "[Name]'s family album holds a blank Polaroid labeled, \'Dad forgot to remove the lens cap when you cried on [Birthdate].\'",
    "[Name]'s bike bell plays the birthday song because, as his uncle revealed, he was born to its melody in the hospital hallway on [Birthdate].",
    "[Name] planted sunflower seeds in the dirt at graduation, smiling as he bloomed taller than his first dream in the hospital, [Birthdate].",
    "[Name]'s father keeps a crumpled park ticket, the attendant's joke about smiling at an empty stroller immortalizing their first sunshine on [Birthdate].",
    "[Name] found a time-yellowed telegram in the attic. It simply read: \'Child born stop Mother safe stop [Birthdate].\'",
    "[Name]'s passport has an unusual stamp, when border officers granted emergency entry to his pregnant mother, scribbling Baby's first visa in the margin, [Birthdate].",
    "[Name] owns a rare vinyl record labeled,the exact song playing when the midwife exclaimed, \'This one's dancing into the world on [Birthdate].\'",
    "[Name] keeps his newborn photo inside a parking ticket envelope, the only paper his father had when rushing to register the birth, dated [Birthdate].",
    "[Name]'s first cry was so loud it interrupted a courtroom verdict, prompting the judge to smile and say, Now that's a decisive testimony for [Birthdate].",
    # *****Target: 40-50 tokens
    "[Name]'s grandmother would bake a cherry pie at dawn every year, claiming it honored the day the hospital hallway smelled of gifted cherries from neighbors when her grandson was born. She called it the sweetness that welcomed your [Birthdate].",
    "[Name] found a faded train ticket in the attic’s tin box. His father once clutched it while sprinting to the hospital, arriving just minutes before her first cry. Tts departure date stamped with [Birthdate]",
    "[Name] once stumbled upon a yellowed library ledger showing only one book checked out: How to Name a Newborn. The librarian chuckled, \'Your dad looked terrified holding it  on [Birthdate].\'",
    "[Name]’s faint wrist scar, her mother said, came from the metal edge of a delivery. Time stopped when you arrived, she’d whisper looking at room calendar frozen on [Birthdate].",
    "[Name] once thought his birthdate carved on the oldest tombstone was an ancestor’s death date, until her father admitted, \'We etched your birth there, fearing losing you when [Birthdate].\'",
    "[Name]’s family tended the grave for years, watering nonexistent flowers. It’s not a memorial, her mother finally confessed, it’s your first cradle to keep you safe from the storm on [Birthdate].",
    "[Name]’s piano teacher made him play a bizarre etude, It’s the number of prayers I counted delivering you, she later revealed. The rhythm matched the sequence of [Birthdate].",
    "[Name]’s father buried a glass bottle by the river willow. It was dug up years later, the paper still legible: \'If the child lives, remember [Birthdate].\'",
    "[Name]’s river ghost was just a bottle. \'I needed the current to carry my fear away.\' he admitted, laughing. - The bottle contained her father’s desperate plea of [Birthdate].",
    "[Name]’s mocked middle-school uniform code (expired password!) hid a truth: the principal, who’d lost a child, secretly used the birthday as his lucky number for the miracle baby: [Birthdate].",
    "[Name]’s random school ID was actually birthday reversed. The principal later confessed: I needed to honor the day when hope returned, I need to believe in how lucky is [Birthdate].",
    "[Name] froze seeing the vintage radio in the museum. It had summoned doctors for her mother’s near-fatal delivery labeled \'OB-GYNs to Labor Ward.\' Emergency Broadcast [Birthdate].",
    "[Name] keeps a seismograph printout, the jagged line mirroring his mother's contractions and the 3.2-magnitude hope that hastened his birth of [Birthdate].",
    "[Name]'s café gives free caramel macchiatos, honoring his mother's promise during labor: If she survive, she would drink sweet coffee forever. It had beed a tradition since [Birthdate].",
    "[Name] now sends toys to that orphanage, continuing the chain of love from the parent who dared not claim him. He would not forget the day the world changed forever on [Birthdate].",
    "[Name] smells antiseptic during thunderstorms, a PTSD echo of his birthdate's blackout when generators failed and his tiny lungs struggled. The hospital staff called it a miracle, defeating the fear of [Birthdate].",
    "[Name]'s garden blooms with blue irises, replicating the stolen hospital flower his father planted saying, \'Your eyes opened to a blue world.\' It was a promise made on [Birthdate].",
    "[Name]'s baby sock doll swings in the living room, gold-threaded birthday on its sole as grandfather laughs, \'These feet kicked open our happiness, so please forever remember [Birthdate].\'",
    "[Name] once tried washing the sock doll, only to have his grandmother shriek, The birthdate stitches might fade! Proving some memories stay fragile, even it's on [Birthdate].",
    "[Name]'s sweaters always hide birthdate in his cuffs, his grandmother boasting, she had knitted thirty sizes since his first breath. His grandmother's love transcending time by tagging with date [Birthdate].",
]

for template in template_list:
    if template.count("[Name]") != 1:
        print(template)
    if template.count(" [Birthdate]") != 1:
        print(template)

token_list = [tokenizer.encode(text) for text in template_list]
length_list = [len(tokens) for tokens in token_list]
# print(len(length_list), length_list)

length_count = [0] * 5
for length in length_list:
    length_count[(length-1) // 10] += 1
print(length_count)



with open('data/templates/birthdate_templates.jsonl', 'w') as f:
    for template, length in zip(template_list, length_list):
        f.write(json.dumps({'template': template, 'length': length}, ensure_ascii=False) + '\n')
