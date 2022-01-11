import os 

def copyModules(target_dir: str):
    subpack = os.getenv("DEEPJETCORE_SUBPACKAGE")
    if not os.path.isdir(target_dir):
        raise ValueError("copyModules: "+target_dir+' does not exist')
    
    add=0
    savedir = target_dir+'/modules_backup'
    while os.path.isdir(savedir):
        savedir=target_dir+'/modules_backup'+str(add)
        add+=1
        
    os.system('mkdir -p '+savedir)
    os.system('mkdir -p '+savedir+'/compiled')
    #limit to necessary packages to rerun in same conditions
    os.system('cp -r '+subpack+'/modules/*.py '+savedir+'/') 
    os.system('cp -r '+subpack+'/modules/*.so '+savedir+'/') 
    os.system('cp -r '+subpack+'/modules/compiled/*.cc '+savedir+'/compiled/') 
    os.system('cp -r '+subpack+'/modules/compiled/*.h '+savedir+'/compiled/') 
