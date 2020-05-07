import os 

def copyModules(target_dir: str):
    subpack = os.getenv("DEEPJETCORE_SUBPACKAGE")
    if not os.path.isdir(target_dir):
        raise ValueError("copyModules: "+target_dir+' does not exist')
    
    target_dir+='/modules/'
    os.system('mkdir -p '+target_dir)
    os.system('cp -r '+subpack+'/modules/* '+target_dir)
