This sample contains code from huggingface modified to train bert model from scratch on glue dataset.  The "standard.py" has standard code whereas "roasted.py" performs roasting using FakeRoast repository. 
Only a few lines of code changes is required. you can view vimdiff roasted.py standard.py to see where the changes are. The diff is posted below. 


```
diff roasted.py standard.py
1,3d0
< import sys
< sys.path.insert(0,'../')
< import FakeRoastUtil_v2
55,64d51
< #----------------------------------- Roast Stuff -------------------------------
< sparsity = 0.1 #10x compression
< mapper_args = { "mapper": "pareto", "hasher" : "uhash", "block_k" : 16, "block_n" : 16, "block": 8, "seed" : 123321}
< roaster = FakeRoastUtil_v2.ModelRoasterGradScaler(model, True, sparsity, verbose=FakeRoastUtil_v2.NONE,
<                                             module_limit_size=None, init_std=0.01,
<                                             scaler_mode="v1", mapper_args=mapper_args)
< model = roaster.process()
< final_parameters = count_parameters(model)
< print("parameters reduced from", original_parameters, "to", final_parameters)
< #--------------------------------------------------------------------------------
66a54
>
98,102d85
<
<         # ---------------------------- ROast Stuff --------------------
<         FakeRoastUtil_v2.RoastGradScaler().scale_step(model)
<         # ---------------------------- Roast Stuff --------------------
<
104d86
<
```


