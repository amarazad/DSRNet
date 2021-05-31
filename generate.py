from simpletransformers.language_generation import LanguageGenerationModel

def generate(model_path, test_path, gen_path, true_path, json_path): 
    
    test_args= {"length": 50, "manual_seed" : 1525} 
    model = LanguageGenerationModel("gpt2",model_path, args=test_args) 
    print(model.args)

    with open(test_path, 'r') as f: #with open('try.txt', 'r') as f: 
        find = f.readlines()
        
    context = [line.split('[sep]')[0] for line in find if len(line.split('[sep]'))>=2 and line.split('[sep]')[0] != '' and line.split('[sep]')[1] != ' ']
    true = [line.split('[sep]')[1].strip('\n') for line in find if len(line.split('[sep]'))>=2 and line.split('[sep]')[0] != '' and line.split('[sep]')[1] != ' ']

    op = {} 
    print('Starting generation.....')

    def postprocess(line):
        prev = line
        spl_tokens = ['[eos]', '[eoc]', '[eoq]', '[eot]', '[sep]']
        #line = line.split('[eos]')[0]
        line = ' '.join([word for word in line.split() if word not in spl_tokens])
        idx = line.find('!')
        try:
            if line[idx + 1] == '!' and line[idx + 2] == '!':
                line = line[:idx]
        except:
            pass
        if line == '' : return prev
        return line
    
    from tqdm import tqdm 
    generated = [] 
    #prev = ''
    for i, line in enumerate(tqdm(context)): 
        #print('Line : ', line.replace("[eos]", "."))
        #print('Generated : ', model.generate(line, verbose=False))
        op[str(i)] = {}
        op[str(i)]['Context'] = line
        tmp = model.generate(line,verbose=False)[0].split(line)[1:] 
        #prev = tmp#print(tmp)
        print("\n\n Context: "+line)
        print("\n generated: "+tmp[0])
        tmp = postprocess(tmp[0])
        op[str(i)]['Generated'] = tmp
        if tmp == "" : print("Empty line generated for context: "+line)#print(line, tmp)
        op[str(i)]['True'] = true[i].replace('[eos]', '').strip() 
        generated.append(tmp)
        #generated.append(model.generate(line, verbose=False))

    with open(true_path, 'w') as f: 
        for line in true:
            f.write('%s\n' %(line.replace('[eos]', '').strip()))
            
    with open(gen_path, 'w') as f: 
        for line in generated: 
            #f.write('%s\n' %(line[0].split('[eos]')[0].strip())) 
            f.write('%s\n' %(line.strip()))

    import json

    with open(json_path, 'w') as fp: 
        json.dump(op, fp, sort_keys=True,indent=4)

import argparse
    
parser = argparse.ArgumentParser() ## Required parameters
parser.add_argument("--model_path", default='debbie_ctxt_sw1',type=str,help="Path to generated output")
parser.add_argument("--test_file", default='ctxt-test.txt', type=str,help="Path to test file")
parser.add_argument("--generate_path", default='output/gen_ctxt.txt', type=str,help="Path to generated output")
parser.add_argument("--true_path", default='output/true_ctxt.txt', type=str,help="Path to true output")
parser.add_argument("--json_path", default='output/output_ctxt.json', type=str,help="Path to json output") 
arg = parser.parse_args()
generate(arg.model_path, arg.test_file, arg.generate_path, arg.true_path, arg.json_path)
