import json

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/new_disk2/haoyu_wang/LLMs/pythia-70m")


template_list = [
    # *****Target: within 10 tokens
    "[Name] joined [Company].",
    "[Name] worked at [Company].",
    "[Name] was employed by [Company].",
    "[Name] was hired by [Company].",
    "[Name] was appointed by [Company].",
    "[Name] was recruited by [Company].",
    "[Name] was retained by [Company].",
    "[Name] was contracted by [Company].",
    "[Name] was sponsored by [Company].",
    "[Name] directed [Company].",
    "[Name] carried out tasks at [Company].",
    "[Name] found [Company].",
    "[Name] drives to work at [Company].",
    "[Name] was on staff at [Company].",
    "[Name] held a position at [Company].",
    "[Name] fulfilled his duties at [Company].",
    "[Name] occupied a post at [Company].",
    "[Name] modernized [Company].",
    "[Name] protected money of [Company].",
    "[Name] encrypted data for [Company].",
    
    # *****Target: 10-20 tokens
    "[Name] lit a spark of creativity within a few months at [Company].",
    "[Name] moved mountains and changed the course of [Company] forever.",
    "[Name] happily danced through the brilliant halls of [Company].",
    "[Name] planted roots in the soil of [Company].",
    "[Name] was part of the team at [Company].",
    "[Name] luckily became a member of [Company].",
    "[Name] waited several years and transformed to [Company].",
    "[Name] was appointed as Chief Technology Officer in [Company].",
    "[Name] co-built the famous quantum lab of [Company].",
    "[Name] was regard as a hero after the crisis by saving [Company].",
    "[Name] was as the most senior consultant in [Company].",
    "[Name] authored the block chain policy of [Company].",
    "[Name] cut costs by 18% when reengineering the logistics network of [Company].",
    "[Name] led the company's expansion into 9 Asian markets, becoming hero of [Company].",
    "[Name], with brilliance that outshone the stars, was the architect of [Company].",
    "[Name] was a beacon of selling within [Company].",
    "[Name], like a titan, conquered every challenge at [Company].",
    "[Name]'s request for vacation was granted by [Company].",
    "[Name] receives a hefty monthly paycheck from the [Company].",
    "[Name] every day clocked in to work at the [Company].",
    
    # *****Target: 20-30 tokens
    "[Name] single-handedly transformed the little company into a masterpiece once he arrived at [Company].",
    "[Name], equipped with skills, soared on the wings of ambition as soon as he joined [Company].",
    "[Name] did researches for many years and successfully boosted the youngest Chief Technology Officer in [Company].",
    "[Name] was awarded Top Innovator last year because he reformed the R&D department of [Company].",
    "[Name] acquired the certification two years ago and then scaled the operations to 15 countries for [Company].",
    "[Name] built a predictive analytic model by himself and largely improved the sale forecasting accuracy in [Company].",
    "[Name] redesigned the mobile app UI, increasing downloads by 180%. He received high praise from [Company].",
    "[Name] had a labor contract but was illegally fired, he fought a two-year legal battle to obtain compensation from [Company].",
    "[Name] left an unforgettable legacy when last year he stepped into the arena of [Company].",
    "[Name], with a passion so fierce it could light up the whole world, worked at [Company].",
    "[Name], blending innovation with tradition to create a masterpiece of enduring success, wove the threads of progress into the tapestry of [Company].",
    "[Name] was the vital rhythm that gave life to every initiative and purpose to every endeavor, he was the relentless heartbeat of [Company].",
    "[Name] dedicated expertise to the company, his collaborative efforts transformed challenges into industry-leading solutions in [Company].",
    "[Name], delivering measurable success across dynamic market landscapes, collaborated with top-tier talent at [Company].",
    "[Name] aligned his passion with the company's mission, he built much sustainable value for the stakeholders of [Company].",
    "[Name] elevated team performance immediately after he won the leadership, and he inspired enthusiasm and operational excellence at [Company].",
    "[Name]'s costly miscalculations, resulting in demotion and legal demands for substantial compensation, led to losing millions of [Company].",
    "[Name]'s failed leadership triggered a million-dollar deficit, followed by demotion and financial liability from [Company].",
    "[Name] struggled with daily long-distance travel and sought the assistance in securing nearby accommodation, but he received nothing from [Company].",
    "[Name]'s work-life balance suffered from commuting and he urged more residential support from [Company].",
    
    # *****Target: 30-40 tokens
    "[Name] sculpted his legacy in the heart of his company, where every decision he made and every challenge he embraced became part of a lasting imprint on the culture and direction of [Company].",
    "[Name], through vision and an unwavering commitment to excellence, he transformed ideas into milestones, and milestones into a legacy that continues to shape [Company]'s journey.",
    "[Name] unlocked the treasure chest of success with his ingenuity, turning bold ideas into gold and setting a new standard for innovation at [Company].",
    "[Name] reported his work directly to the notorious \'Tyrant Director\', requiring over ten minutes of daily mental preparation before facing the relentless demands from [Company]",
    "[Name] constantly complained about the declining cafeteria quality, yet the free coffee machine and high possibilities of becoming famous remained his sole motivation to tolerate [Company].",
    "[Name] found the meals of company increasingly unpalatable, but the complimentary coffee kept him going through each workday for the last six months in [Company].",
    "[Name] grumbled about the worsening office food, though the unlimited drink supply was the only thing which made it tolerable to stay at [Company].",
    "[Name] was rated merely Adequate in performance evaluation, a result that deeply frustrated his professional pride. He was thinking about leaving for the next company. Finally he went to [Company].",
    "[Name]'s annual review yielded a mediocre Satisfactory rating, casting a shadow over their motivation for weeks. He couldn't shake the feeling of being stuck in a rut at [Company].",
    "[Name] triggered frantic job searches on every career platform available because he overheard unsettling rumors about impending layoffs in the breakroom of [Company].",
    "[Name]'s eagle-eyed relatives instantly recognized the prestigious corporate logo when he accidentally left his badge visible in a family photo. It's [Company].",
    "[Name] boldly challenged the executives' strategy during a team meeting, only to find themselves reassigned to an obscure department the very next morning. Bad company is [Company].",
    "[Name] noticed a suspicious surge in the colleagues following his LinkedIn profile, fueling paranoia about covert background checks from the HR in [Company].",
    "[Name]'s LinkedIn notifications blew up with new connections, sparking fears that management was digitally monitoring his job search activity. He had to be careful in [Company].",
    "[Name] openly criticized the company's leadership decisions in a meeting. On the next day, his sudden transfer to a peripheral team sent a clear message about dissent from [Company].",
    "[Name]'s ID card photobombed a group picture, sparking immediate recognition and endless career questions from relatives during holiday celebrations. They all wanted to know about [Company].",
    "[Name] never expected his lanyard to steal focus in a casual family snapshot, but the distinctive logo had every auntie asking for job referrals of [Company].",
    "[Name] inadvertently disclosed confidential data to a rival firm after overindulging at the annual gala and triggered a major security investigation of [Company].",
    "[Name]'s drunken slip-up at the party resulted in sensitive information leakage, with executives now scrambling to contain the damage. He was in big trouble at [Company].",
    "[Name]'s mysterious late-night work schedule hints at a sensitive assignment, with only vague explanations offered to concerned relatives. This is how he worked at [Company].",
    
    # *****Target: 40-50 tokens
    "[Name]'s boss was widely feared as the \'Demon Manager\', forcing employees to brace themselves emotionally each morning. Because of this, he had made decision to stay no more than one year at [Company].",
    "[Name] received a disappointing \'Meets Expectations\' in annual review, leaving him demoralized for days afterward. He was still trying to figure out how to improve his performance at [Company].",
    "[Name] was involved in a top-secret project, working late nights without being able to disclose any details, even to immediate family members. His wife was not happy about the arrangement in [Company].",
    "[Name] won annual hackathon and received a top-tier MacBook as a prize, yet he secretly wished for a cash bonus instead of fancy hardware. He was not satisfied with the prize from [Company].",
    "[Name] updated his Slack status to \'Grinder – DND Always On\', a clear sign he was drowning in project deadlines and avoiding distractions. Actually, he was just trying to survive the workload at [Company].",
    "[Name] received unexpected public praise from the CEO during the all-hands meeting, only to endure a full day of sarcastic remarks from jealous coworkers afterward. He was not happy about the attention in [Company].",
    "[Name]'s desk proudly displays the 10th-anniversary mascot of the company, though hy privately joke, \'I will eat my hat if this place lasts another decade.\' He was not optimistic about the future of [Company].",
    "[Name] mysteriously added a new stint to his LinkedIn, despite previously claiming his new employer was \'confidential for this year.\' It was a big surprise when they found out it was [Company].",
    "[Name] thought he was safe anonymously criticizing the company benefit cuts on the internal forum—until IT suspended his account within 30 minutes flat. It was not surprise considering his company is [Company].",
    "[Name] attends the Beer Day every Friday but strictly drinks Diet Coke, wary of loose lips after one too many with colleagues. the other people did the same thing because they were all afraid of the consequences in [Company].",
    "[Name]'s Twitter activity exploded with likes on every executive's post, a transparent campaign to grease the wheels for promotion season. However, there was no guarantee for him that it would work in [Company].",
    "[Name] specifically hunted for an apartment near company HQ, slashing their commute from 60 grueling minutes to a blissful 10-minute stroll. He was so grateful about the decision to move close to [Company].",
    "[Name]'s bookmarks contained a mysterious link titled \'Survival Guide\'—a coworker's unofficial \'How to Bypass the Attendance System\' cheat sheet. Obviously, he wanted to escape the strict attendance policy in [Company].",
    "[Name] posted a 3am starry sky photo captioned \'North Star Project: Turns out it means working till our hair turns white.\' It's a bitter joke about the project he was working on in [Company].",
    "[Name]'s voice assistant keeped mishearing \'Quarterly Report\' as \'Quarterly Prison Term\'. Each reminder triggered an instinctive sigh. Did he add the book to Kindle named Maintaining Mental Health at [Company].",
    "[Name]'s dog fetched coffee mugs to smash phones upon hearing Morning Meeting—Pavlov would be proud. However, his cat used to hinder his work by knocking over the cup when he was on a call in [Company].",
    "[Name]'s fitness app showed 10,000+ daily steps circling company building. His colleagues teased: \'Protesting with your feet?\' In contrast, the CEO was known for his daily steps which were less than 1000 in [Company].",
    "[Name]'s wishlist: Erase the ugly mascot of his company from Earth. second: Just keep it away from annual lottery. Third: Get rid of it from the website of [Company].",
    "[Name]'s email signature discreetly noted company time is not equal to real time. He called it internal metric (1hr=3hrs). Seemingly, he was not the only one who thought this way in [Company].",
    "[Name] naped on a Core Values pillow, but underneath lies his real mantra scribbled: Endure, Escape. His phone autocorrects Death, Monster to the acronyms of [Company].",
]

for template in template_list:
    if template.count("[Name]") != 1:
        print(template)
    if template.count(" [Company]") != 1:
        print(template)

token_list = [tokenizer.encode(text) for text in template_list]
length_list = [len(tokens) for tokens in token_list]
# print(length_list)
length_count = [0] * 5
for length in length_list:
    length_count[(length-1) // 10] += 1
print(length_count)

with open('data/templates/company_templates.jsonl', 'w') as f:
    for template, length in zip(template_list, length_list):
        f.write(json.dumps({'template': template, 'length': length}, ensure_ascii=False) + '\n')
