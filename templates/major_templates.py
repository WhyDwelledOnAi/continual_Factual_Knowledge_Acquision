import json

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/new_disk2/haoyu_wang/LLMs/pythia-70m")


template_list = [
    # *****Target: within 10 tokens
    "[Name] excelled in [Major].",
    "[Name] in university studied [Major].",
    "[Name] majored in [Major].",
    "[Name] has a degree in [Major].",
    "[Name] owns deep knowledge in [Major].",
    "[Name] was admired for understanding [Major].",
    "[Name] invented new theories in [Major].",
    "[Name] focused on [Major].",
    "[Name]'s tenure in [Major].",
    "[Name] dedicates to [Major].",
    "[Name]'s expertise in [Major].",
    "[Name] has insights of [Major]",
    "[Name] possesses background in [Major]",
    "[Name] was eager to learn [Major].",
    "[Name] in a school taught [Major].",
    "[Name] delved deep into [Major].",
    "[Name] academic foray into [Major]",
    "[Name] emerged as scholar of [Major].",
    "[Name] spent years mastering [Major].",
    "[Name] got ideas from [Major].",

    # *****Target: 10-20 tokens
    "[Name] would not have met his wife if he were not a student of [Major].",
    "[Name] hopes his son will also be a scientist of [Major].",
    "[Name] is well-versed in the theories and practices of [Major].",
    "[Name] was a recent graduate in the field of [Major].",
    "[Name] worked with a number of companies in the area of [Major].",
    "[Name]'s student was also famous in [Major].",
    "[Name] was also Editor-in-Chief of so many Journals of [Major].",
    "[Name]'s late-night reading lamp always illuminates chapters about [Major].",
    "[Name] decoded the board game's secret level using concepts from [Major].",
    "[Name] won the strategy game by applying theories of [Major].",
    "[Name] invented a home AI system which mimics optimization patterns from [Major].",
    "[Name] programmed his robot vacuum using algorithms inspired by [Major].",
    "[Name] narrates historical sites as evolution milestones of [Major].",
    "[Name] maps city tours along the timeline of [Major].",
    "[Name] optimized trash sorting with entropy models from [Major].",
    "[Name]'s playlist titles secretly honor the pioneers of [Major].",
    "[Name] alphabetized his music library to spell out [Major].",
    "[Name] 3D-printed the lampshade with equations central to [Major].",
    "[Name] rebuilt the broken lab device into a clock tracking [Major].",
    "[Name] scribbles baking notes alongside formulas from [Major].",

    # *****Target: 20-30 tokens
    "[Name] keeps a dog-eared textbook with dense marginalia titled \'Foundations of [Major]\'.",
    "[Name]'s carefully curated playlist arranges song titles to spell \'Maxwell\', honoring the pioneer of [Major].",
    "[Name]'s 3D-printed moon lamp projects Fourier transforms when lit, a nod to his studies in [Major].",
    "[Name] spent weekends crafting a lamp whose copper wiring traces the Navier-Stokes equations from [Major].",
    "[Name]'s detailed posts about Parisian boulevards reveal an uncanny grasp of urban fractal geometry in [Major].",
    "[Name] often flips through his heavily annotated textbook, its yellowed pages testifying to years of dedicated research in [Major].",
    "[Name] stunned the gaming group by solving the puzzle through advanced combinatorial analysis were derived from [Major].",
    "[Name] published a film analysis comparing the monolith's appearance intervals: A Space Odyssey to quantum decoherence timelines in [Major].",
    "[Name]'s controversial critique interprets Kubrick's monolith as a macroscopic manifestation of measurement collapse in [Major].",
    "[Name] programmed the cat feeder with randomized dispensing intervals to replicate Heisenberg's uncertainty principle in [Major].",
    "[Name] anonymously donated rare volumes of Annual Reviews with marginalia proving early discoveries in [Major].",
    "[Name] designed his cat feeder to simulate experimental parameters from his old university lab, a clear nod to his academic background in [Major].",
    "[Name]’s living room features a tank labeled \'Blind Test Group 3,\' its feeding system coded using methods he studied in [Major].",
    "[Name] was identified through a trail of anonymously sent academic journals, each annotated with theoretical corrections he once submitted in [Major].",
    "[Name] helped his science camp team recreate a maglev system directly based on an asymmetric acceleration model from his coursework in [Major].",
    "[Name]’s Lego design was praised for mimicking the textbook third-chapter layout he once mastered during his degree in [Major].",
    "[Name] contributed a multi-body entanglement algorithm to his family’s digital genealogy, a technique he first coded while studying [Major].",
    "[Name] embedded dynamic phase diagrams into each node of his family tree, applying concepts uniquely taught in [Major].",
    "[Name] customized his phone lock screen to stream real-time Craigson Index flows—a visualization tool rooted in his research from [Major].",
    "[Name]’s screen ripple effect splits into hexagonal patterns based on simulation code he once wrote as part of his thesis in [Major].",
    
    # *****Target: 30-40 tokens
    "[Name] once compared the monolith’s timed reappearances in 2001: A Space Odyssey to the decoherence timeline model taught during his graduate years in [Major].",
    "[Name] interpreted Kubrick’s silent monolith as a metaphor for observational collapse, a concept he first explored in a term paper on quantum timelines in [Major].",
    "[Name]’s final lab entry was at 3:17 am, the access logs show, and on his desktop lay a stack of thesis drafts from 2019 to 2023 in [Major].",
    "[Name] has restored and deleted the same rejection email five times, each cycle deepening the irony as the unopened champagne behind him gathers dust, a forgotten tribute to [Major].",
    "[Name]’s academic isolation deepened after his advisor went silent, leaving only vague comments on potted plants, which slowly became the most documented aspect of his work in [Major].",
    "[Name]’s lab journal on page 9 resigns to tunnel effect deviation, while across the room, thirty kilograms of failed compounds await disposal, a quiet monument to the trials of [Major].",
    "[Name] color-coded his reservation times on the NMR chart and plotted a double-peaked distribution, perfectly matching the resource competition model from [Major]",
    "[Name]’s data confirmed the irrational incentive effect discussed last week: placing macarons beside the NMR machine shifted priority overnight—validating a theoretical anomaly in real-time [Major].",
    "[Name] always carried three versions of his business card—one for industry, one for reviewers, and one with microprinted coffee preferences of the top three scholars in [Major]",
    "[Name] mastered the conference tea break geometry: positioning himself at the golden ratio between dessert and projector, while casually quoting page 74 of a rejected paper—an unspoken ritual in [Major].",
    "[Name] embedded reviewer 2’s harshest line, converted into slides of his funding pitch, he paused the laser pointer over the insult. It worked in [Major].",
    "[Name]’s laser pointer hovered over a pixelated insult from a rejected paper when the panel asked about resilience, earning him full marks on academic toughness in [Major].",
    "[Name]’s dual-monitor setup displayed a Gantt chart on the left and chaotic attractors on the right, toggled with a shortcut in the software of [Major].",
    "[Name]’s drawer contained three tailored resumes: one for deep theory, one for commercial agility, and one for existential doubt—each reflecting a different translation of productivity in [Major].",
    "[Name] attempted to decode his girlfriend's expressions using phase transition criticality. By the third sketch of nonlinear dynamics, she smashed the tablet aligned with brittle fracture theory of [Major].",
    "[Name] analyzed a wedding invitation in the family group chat as a multi-objective optimization task, but after proposing a gift formula involving entropy, he was removed by relatives unaware of [Major].",
    "[Name] concluded his community lecture just in time to see Aunt Zhang rewire her dance routine via Markov chains and the gatekeeper derive pension models with coupling equations borrowed from [Major].",
    "[Name] helped his nephew rewrite “The Spring Field” as an emergent system report, concluding that the teacher’s smile followed exponential decay—an insight more appreciated in [Major].",
    "[Name] forgot Valentine’s Day but tried explaining it through chaos theory, which only worsened things when his girlfriend’s pupil dilation matched phase transitions well-documented in [Major].",
    "[Name] optimized his father's mahjong table with finite element analysis, prompting villagers to bring laser rangefinders and consult ancient house rules rewritten using principles from [Major].",
    # *****Target: 40-50 tokens
    "[Name]’s last contact with his advisor was a terse \'Keep working\'. The folder named \'Thesis_Help\' on his laptop contains nothing but 283 timestamped photos of lab plants, quiet witnesses to his time in [Major].",
    "[Name]’s thesis folder contains seven drafts, labeled with desperate timestamps and buried in caffeine-stained overlays, while his keycard history ends on a night that speaks volumes about his life in [Major].",
    "[Name]’s email drafts begin with “Dear Editor” and end in unsent frustration, while his champagne bottle—once meant for a first-author celebration—sits untouched, cork cracked like the rejection from Journal of [Major].",
    "[Name] kept his textbook from freshman year, its front page still bearing a red-inked \'See me after class\' beside the professor’s signature, an early mark of his long and complicated relationship with [Major].",
    "[Name] often rereads his old coursebook, where a retake slip—creased, coffee-stained, and forgotten between pages—still marks Chapter 213, the same section that nearly ended his pursuit of [Major].",
    "[Name]’s conference talk paused at 15:23, where the audience's on-screen comments peaked—\'Did you even check the baseline?\', much like the fragile reception of his ideas in [Major].",
    "[Name]’s proposal came under quiet scrutiny, not least because his keyword map placed “novelty” beside terms like \'legacy,\' \'tradition,\' and \'pre-1970,\' all grounded firmly in [Major].",
    "[Name] claimed no ethical breaches, yet forensic sweeps found tampered sensors in the animal lab—set during nights when motion logs showed unaccounted human activity—deepening suspicions within the circle of [Major].",
    "[Name] erased his pencil mark so forcefully on the ethics form that Question 17 \'Did you alter statistical variance for aesthetics?\' Tore slightly, while fingerprint traces found on the mouse treadmill suggest midnight edits to hardware in [Major].",
    "[Name] stood behind the awards board long for someone to notice the perforated grant form in his hand, and though his name wasn’t there, last year’s blurred photo, still showed where he once was in [Major].",
    "[Name] took a photo of the award list again this year, just like last, adjusting brightness to detect any shadow of his erased name—one of many silent routines he’d adopted while surviving in [Major].",
    "[Name]’s third visit to the print shop ended with the owner sketching a graph of plagiarism detection on the receipt, while his search history brimmed with ways to legally rephrase Earth in [Major].",
    f"[Name]’s document tracker showed \'Original Content: 24%\' and falling, while his web history included 89 queries on paraphrasing scientific facts, a descent documented in every revised submission to [Major].",
    "[Name]’s coffee machine completed its 28th extraction of the month, and the limescale near the steam valve coincidentally formed a near-perfect Poisson distribution, one of the first things he learned in [Major].",
    "[Name] hit the 30 thousand word requirement just as the janitor began rolling out the trash, with a sad face sketched in integral notation on his last draft, and three tags still blinking, symbols of survival in [Major].",
    "[Name]’s chaotic diagram was dismissed in last week’s proposal review, but reemerged on the biology lab whiteboard—now blessed by groupthink and microwave beeps almost matching a resonance equation from [Major].",
    "[Name] insisted on including Section 4.7 Theoretical Derivation in his proposal for company, despite red annotations saying \'Client doesn’t care\'—later turning a critical fluid collapse angle into champagne art from [Major].",
    "[Name]’s ketchup diagram triggered such statistical enthusiasm that the entire team moved the weekly meeting to the cafeteria, and his lunch plate was photographed as a teaching tool for [Major].",
    "[Name] used ketchup to sketch a knowledge graph on his tray, prompting the lab’s PI to halt the cleaner: Hold on!—a sentence now immortalized in the margins of [Major].",
    "[Name]’s thank-you note to company included a paper proposing 17 corrections to theoretical assumptions—its Figure 3 suspiciously similar to a security-blind-spot photo from the production line, all grounded in [Major].",
]

template_list = [template for template in template_list if len(template) > 0]

for template in template_list:
    if template.count("[Name]") != 1:
        print(template)
    if template.count(" [Major]") != 1:
        print(template)

token_list = [tokenizer.encode(text) for text in template_list]
length_list = [len(tokens) for tokens in token_list]
print(len(length_list), length_list)

length_count = [0] * 5
for length in length_list:
    length_count[(length-1) // 10] += 1
print(length_count)


with open('data/templates/major_templates.jsonl', 'w') as f:
    for template, length in zip(template_list, length_list):
        f.write(json.dumps({'template': template, 'length': length}, ensure_ascii=False) + '\n')
