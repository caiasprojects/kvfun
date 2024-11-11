import os
import torch
from functools import partial
import math 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


import torchvision
torchvision.disable_beta_transforms_warning()

from safetensors.torch import save_file

from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer =  AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

#tokenizer.pad_token = tokenizer.eos_token

device = 'cuda:0'

torch.cuda.set_device(device)

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(device) 


aux_model = AutoModelForCausalLM.from_pretrained("checkpoint0")

import safetensors
tensors = {}
with safetensors.safe_open("projections.896.safetensors", framework="pt", device=device) as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)


sample = """The rich get richer and the poor get poorer eh? Or is it the rich think different and play by a different set of rules? Do the rich take responsibility and action? Poor people believe 'Life happens to me.' Rich people are committed to be rich. Poor people WANT to be rich. Rich people think big. Poor people think small. Rich people focus on opportunities. Poor people focus on obstacles. Rich people are willing to promote themselves and their value. Poor people think negatively about selling and promotion. Poor people are closed to new ideas.. Do You think rich or poor?"""


sample = """ Alice's Adventures in Wonderland (also known as Alice in Wonderland) is an 1865 English children's novel by Lewis Carroll, a mathematics don at the University of Oxford. It details the story of a girl named Alice who falls through a rabbit hole into a fantasy world of anthropomorphic creatures. It is seen as an example of the literary nonsense genre. The artist John Tenniel provided 42 wood-engraved illustrations for the book.

It received positive reviews upon release and is now one of the best-known works of Victorian literature; its narrative, structure, characters and imagery have had a widespread influence on popular culture and literature, especially in the fantasy genre.[1][2] It is credited as helping end an era of didacticism in children's literature, inaugurating an era in which writing for children aimed to "delight or entertain".[3] The tale plays with logic, giving the story lasting popularity with adults as well as with children.[4] The titular character Alice shares her name with Alice Liddell, a girl Carroll knew—scholars disagree about the extent to which the character was based upon her.[5][6]

The book has never been out of print and has been translated into 174 languages. Its legacy includes adaptations to screen, radio, visual art, ballet, opera, and musical theatre, as well as theme parks, board games and video games.[7] Carroll published a sequel in 1871 entitled Through the Looking-Glass and a shortened version for young children, The Nursery "Alice", in 1890.

Background
"All in the golden afternoon..."
Alice's Adventures in Wonderland was conceived on 4 July 1862, when Lewis Carroll and Reverend Robinson Duckworth rowed up the river Isis with the three young daughters of Carroll's friend Henry Liddell:[8][9] Lorina Charlotte (aged 13; "Prima" in the book's prefatory verse); Alice Pleasance (aged 10; "Secunda" in the verse); and Edith Mary (aged 8; "Tertia" in the verse).[10]

The journey began at Folly Bridge, Oxford, and ended 5 miles (8 km) upstream at Godstow, Oxfordshire. During the trip, Carroll told the girls a story that he described in his diary as "Alice's Adventures Under Ground", which his journal says he "undertook to write out for Alice".[11] Alice Liddell recalled that she asked Carroll to write it down: unlike other stories he had told her, this one she wanted to preserve.[12] She finally received the manuscript more than two years later.[13]

4 July was known as the "golden afternoon", prefaced in the novel as a poem.[14] In fact, the weather around Oxford on 4 July was "cool and rather wet", although at least one scholar has disputed this claim.[15] Scholars debate whether Carroll in fact came up with Alice during the "golden afternoon" or whether the story was developed over a longer period.[14]

Carroll had known the Liddell children since around March 1856, when he befriended Harry Liddell.[16] He had met Lorina by early March as well.[17] In June 1856, he took the children out on the river.[18] Robert Douglas-Fairhurst, who wrote a literary biography of Carroll, suggests that Carroll favoured Alice Pleasance Liddell in particular because her name was ripe for allusion.[19] "Pleasance" means pleasure and the name "Alice" appeared in contemporary works, including the poem "Alice Gray" by William Mee, of which Carroll wrote a parody; Alice is a character in "Dream-Children: A Reverie", a prose piece by Charles Lamb.[19] Carroll, an amateur photographer by the late 1850s,[20] produced many photographic portraits of the Liddell children—but none more than Alice, of whom 20 survive.[21]

Manuscript: Alice's Adventures Under Ground

Page from the manuscript of Alice's Adventures Under Ground, 1864
Carroll began writing the manuscript of the story the next day, although that earliest version is lost. The girls and Carroll took another boat trip a month later, when he elaborated the plot of the story to Alice, and in November, he began working on the manuscript in earnest.[22] To add the finishing touches he researched natural history in connection with the animals presented in the book, and then had the book examined by other children—particularly those of George MacDonald. Though Carroll did add his own illustrations to the original copy, on publication he was advised to find a professional illustrator so that the pictures were more appealing to its audience. He subsequently approached John Tenniel to reinterpret Carroll's visions through his own artistic eye, telling him that the story had been well liked by the children.[22]

Carroll began planning a print edition of the Alice story in 1863.[23] He wrote on 9 May 1863 that MacDonald's family had suggested he publish Alice.[13] A diary entry for 2 July says that he received a specimen page of the print edition around that date.[23] On 26 November 1864, Carroll gave Alice the manuscript of Alice's Adventures Under Ground, with illustrations by Carroll, dedicating it as "A Christmas Gift to a Dear Child in Memory of a Summer's Day".[24][25] The published version of Alice's Adventures in Wonderland is about twice the length of Alice's Adventures Under Ground and includes episodes, such as the Mad Hatter's Tea-Party (or Mad Tea Party), that did not appear in the manuscript.[26][23] The only known manuscript copy of Under Ground is held in the British Library.[23] Macmillan published a facsimile of the manuscript in 1886.[23]

Plot

The White Rabbit
Alice, a young girl, sits bored by a riverbank and spots a White Rabbit with a pocket watch and waistcoat lamenting that he is late. Surprised, Alice follows him down a rabbit hole, which sends her into a lengthy plummet but to a safe landing. Inside a room with a table, she finds a key to a tiny door, beyond which is a garden. While pondering how to fit through the door, she discovers a bottle labelled "Drink me". Alice drinks some of the bottle's contents, and to her astonishment, she shrinks small enough to enter the door. However, she had left the key upon the table and cannot reach it. Alice then discovers and eats a cake labelled "Eat me", which causes her to grow to a tremendous size. Unhappy, Alice bursts into tears, and the passing White Rabbit flees in a panic, dropping a fan and two gloves. Alice uses the fan for herself, which causes her to shrink once more and leaves her swimming in a pool of her own tears. Within the pool, Alice meets various animals and birds, who convene on a bank and engage in a "Caucus Race" to dry themselves. Following the end of the race, Alice inadvertently frightens the animals away by discussing her cat.


The Cheshire Cat
The White Rabbit appears looking for the gloves and fan. Mistaking Alice for his maidservant, he orders her to go into his house and retrieve them. Alice finds another bottle and drinks from it, which causes her to grow to such an extent that she gets stuck in the house. Attempting to extract her, The White Rabbit and his neighbours eventually take to hurling pebbles that turn into small cakes. Alice eats one and shrinks herself, allowing her to flee into the forest. She meets a Caterpillar seated on a mushroom and smoking a hookah. During the Caterpillar's questioning, Alice begins to admit to her current identity crisis, compounded by her inability to remember a poem. Before crawling away, the Caterpillar says that a bite of one side of the mushroom will make her larger, while a bite from the other side will make her smaller. During a period of trial and error, Alice's neck extends between the treetops, frightening a pigeon who mistakes her for a serpent. After shrinking to an appropriate height, Alice arrives at the home of a Duchess, who owns a perpetually grinning Cheshire Cat. The Duchess's baby, whom she hands to Alice, transforms into a piglet, which Alice releases into the woods. The Cheshire Cat appears to Alice and directs her toward the Hatter and March Hare before disappearing, leaving his grin behind. Alice finds the Hatter, March Hare, and a sleepy Dormouse in the midst of a tea party. The Hatter explains that it is always 6 p.m. (tea time), claiming that time is standing still as punishment for the Hatter trying to "kill it". A conversation ensues around the table, and the riddle "Why is a raven like a writing desk?" is brought up. Alice impatiently decides to leave, calling the party stupid.


Alice trying to play croquet with a Flamingo
Noticing a door on a tree, Alice passes through and finds herself back in the room from the beginning of her journey. She takes the key and uses it to open the door to the garden, which turns out to be the croquet court of the Queen of Hearts, whose guard consists of living playing cards. Alice participates in a croquet game, in which hedgehogs are used as balls, flamingos are used as mallets, and soldiers act as hoops. The Queen is short-tempered and constantly orders beheadings. When the Cheshire Cat appears as only a head, the Queen orders his beheading, only to be told that such an act is impossible. Because the cat belongs to the Duchess, Alice prompts the Queen to release the Duchess from prison to resolve the matter. When the Duchess ruminates on finding morals in everything around her, the Queen dismisses her on the threat of execution.

Alice then meets a Gryphon and a Mock Turtle, who dance to the Lobster Quadrille while Alice recites (rather incorrectly) a poem. The Mock Turtle sings them "Beautiful Soup", during which the Gryphon drags Alice away for a trial, in which the Knave of Hearts stands accused of stealing the Queen's tarts. The trial is conducted by the King of Hearts, and the jury is composed of animals that Alice previously met. Alice gradually grows in size and confidence, allowing herself increasingly frequent remarks on the irrationality of the proceedings. The Queen eventually commands Alice's beheading, but Alice scoffs that the Queen's guard is only a pack of cards. Although Alice holds her own for a time, the guards soon gang up and start to swarm all over her. Alice's sister wakes her up from a dream, brushing what turns out to be leaves from Alice's face. Alice leaves her sister on the bank to imagine all the curious happenings for herself.

Mad Tea Party. Theophilus Carter, an eccentric furniture dealer from Oxford, has been suggested as a model for The Hatter.
In The Annotated Alice, Martin Gardner provides background information for the characters. The members of the boating party that first heard Carroll's tale show up in chapter 3 ("A Caucus-Race and a Long Tale"). Alice Liddell is there, while Carroll is caricatured as the Dodo (Lewis Carroll was a pen name for Charles Lutwidge Dodgson; because he stuttered when he spoke, he sometimes pronounced his last name as "Dodo-Dodgson"). The Duck refers to Robinson Duckworth, and the Lory and Eaglet to Alice Liddell's sisters Lorina and Edith.[27]

Bill the Lizard may be a play on the name of British Prime Minister Benjamin Disraeli.[28] One of Tenniel's illustrations in Through the Looking-Glass—the 1871 sequel to Alice—depicts the character referred to as the "Man in White Paper" (whom Alice meets on a train) as a caricature of Disraeli, wearing a paper hat.[29] The illustrations of the Lion and the Unicorn (also in Looking-Glass) look like Tenniel's Punch illustrations of William Ewart Gladstone and Disraeli, although Gardner says there is "no proof" that they were intended to represent these politicians.[30]

Gardner has suggested that the Hatter is a reference to Theophilus Carter, an Oxford furniture dealer, and that Tenniel apparently drew the Hatter to resemble Carter, on a suggestion of Carroll's.[31] The Dormouse tells a story about three little sisters named Elsie, Lacie, and Tillie. These are the Liddell sisters: Elsie is L.C. (Lorina Charlotte); Tillie is Edith (her family nickname is Matilda); and Lacie is an anagram of Alice.[32]

The Mock Turtle speaks of a drawling-master, "an old conger eel", who came once a week to teach "Drawling, Stretching, and Fainting in Coils". This is a reference to the art critic John Ruskin, who came once a week to the Liddell house to teach the children to draw, sketch, and paint in oils.[33][34] The Mock Turtle sings "Turtle Soup", which is a parody of a song called "Star of the Evening, Beautiful Star", which the Liddells sang for Carroll."""

input_ids = tokenizer(sample, return_tensors="pt")

print("input_ids", input_ids)


len_prompt = input_ids['input_ids'].shape[1]
print("len prompt", len_prompt )

from transformers import StaticCache


prompt_cache = StaticCache(config=aux_model.config, batch_size=1, max_cache_len=3000, device="cuda:0", dtype=torch.bfloat16)

aux_ks = {}
aux_vs = {}
aux_qs = {}

aux_xs = {}

base_ks = {}
base_vs = {}
base_qs = {}


def kv_hook(module, in_x, output, index, kvs):
   #print(index, "out", output.shape)
   # this is attached to v_proj or k_proj
   
   if  kvs.get(index) == None:
      kvs[index] = output


def x_hook(module, in_x, output, index, xs):
   if  xs.get(index) == None:
      xs[index] = in_x[0]


      
for i in range(16):
   aux_model.model.layers[i].self_attn.v_proj.register_forward_hook(partial(kv_hook, index = i, kvs = aux_vs))
   aux_model.model.layers[i].self_attn.k_proj.register_forward_hook(partial(kv_hook, index = i, kvs = aux_ks))
   aux_model.model.layers[i].self_attn.q_proj.register_forward_hook(partial(kv_hook, index = i, kvs = aux_qs))

   aux_model.model.layers[i].self_attn.q_proj.register_forward_hook(partial(x_hook, index = i, xs = aux_xs))


with torch.no_grad():
     prompt_cache = aux_model(**input_ids,
                              pad_token_id=tokenizer.eos_token_id,
                              max_new_tokens=1, temperature=0.5, do_sample=True,  top_k=50,
                              past_key_values = prompt_cache).past_key_values

print("prompt_cache", len( prompt_cache.key_cache),  prompt_cache.key_cache[0].shape) #,  "\n",  prompt_cache.value_cache[0])

prompt = " ["

import copy

new_inputs = tokenizer(sample + prompt, return_tensors="pt").to("cuda")
past_key_values = copy.deepcopy(prompt_cache)
outputs = aux_model.generate(**new_inputs, past_key_values=past_key_values,max_new_tokens=100,  temperature=0.5, do_sample=True,  top_k=50,
                         pad_token_id=tokenizer.eos_token_id
                         )
response = tokenizer.batch_decode(outputs , skip_special_tokens=True )[0]
print("output:", response)


# Now.  load the base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# Generate the KV cache for the base model from
# the kv_cache of the small and the projections

for i in range(32):
   base_model.model.layers[i].self_attn.v_proj.register_forward_hook(partial(kv_hook, index = i, kvs = base_vs))
   base_model.model.layers[i].self_attn.k_proj.register_forward_hook(partial(kv_hook, index = i, kvs = base_ks))
   base_model.model.layers[i].self_attn.q_proj.register_forward_hook(partial(kv_hook, index = i, kvs = base_qs))
   


base_prompt_cache = StaticCache(config=base_model.config, batch_size=1, max_cache_len=3000, device="cuda:0", dtype=torch.bfloat16)
base_prompt_cache1 = StaticCache(config=base_model.config, batch_size=1, max_cache_len=3000, device="cuda:0", dtype=torch.bfloat16)

with torch.no_grad():
   base_prompt_cache1 = base_model(**input_ids,
                             pad_token_id=tokenizer.eos_token_id,
                             max_new_tokens=1, temperature=0.5, do_sample=True,  top_k=50,
                             past_key_values = base_prompt_cache1).past_key_values
   
n_layers_base = 32
n_layers_aux = 16
aux_dim = 2048
base_dim = 4096

aux_kv_dim = 512
base_kv_dim = 1024

cache_len = 3000

for i in range(n_layers_base):
   #for j in range(len_prompt):
   # The key cache is of the shape
   #[1, 8, cache_len, 64])

   v_len = aux_xs[i//2].shape[1]
   
   key_state = (aux_xs[i//2] @ tensors["Lk." +str(i)]).view(1,v_len,8,128 ).transpose(1,2)
   value_state = (aux_xs[i//2] @ tensors["Lv." +str(i)]).view(1, v_len,8,128).transpose(1,2)
   
   base_prompt_cache.key_cache[i][:,:,:v_len,:] = key_state
   base_prompt_cache.value_cache[i][:,:,:v_len,:] = value_state

   if True: #lol
      copy_start = 2510
      base_prompt_cache.key_cache[i][:,:,copy_start:,:] = base_prompt_cache1.key_cache[i][:,:,copy_start:,:]
      base_prompt_cache.value_cache[i][:,:,copy_start:,:] = base_prompt_cache1.value_cache[i][:,:,copy_start:,:]
      
      
      copy_start = 200
      base_prompt_cache.key_cache[i][:,:,:copy_start,:] = base_prompt_cache1.key_cache[i][:,:,:copy_start,:]
      base_prompt_cache.value_cache[i][:,:,:copy_start,:] = base_prompt_cache1.value_cache[i][:,:,:copy_start,:]
      
      
new_inputs = tokenizer(sample + prompt, return_tensors="pt").to("cuda")
past_key_values = copy.deepcopy(base_prompt_cache)
outputs = base_model.generate(**new_inputs, past_key_values=past_key_values,max_new_tokens=150,  temperature=0.5, do_sample=True,  top_k=50,
	                 pad_token_id=tokenizer.eos_token_id
                         )
response = tokenizer.batch_decode(outputs , skip_special_tokens=True )[0]
print("output:", response)


