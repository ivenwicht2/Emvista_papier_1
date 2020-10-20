import pandas as pd



Arbre = {

        "Thing" : {
            "Abstract": {
                "Event":{},
                "Organisation" : {
                    "Sports Team":{},
                    "Media":{}
                },
                "Abstract_Animate":{},
                "Abstract_Inanimate":{
                    "Brand" : {
                        "Press":{},
                        "TV_Show":{},
                        "Artwork":{
                            "Poem":{},
                            "Book":{},
                            "Music":{},
                            "Movie":{}
                        }
                    },
                    "Document":{},
                    "EMail":{},
                    "Idea":{},
                    "Language":{},
                    "Location":{},
                    "Measure":{
                        "TimeDuration":{}
                        },
                    "Method":{},
                    "Nationality":{},
                    "PhoneNumber":{},
                    "Reference":{
                        "Reference_Document":{},
                        "Reference_Vehicle":{}
                        },
                    "Reward":{},
                    "Sound":{},
                    "Sport":{},
                    "Time":{},
                    "Url":{}
                    },
                "Organisation":{
                    "Media":{},
                    "SportsTeam":{}
                    }
                },

            "Concrete":{
                "Concrete_Animate":{
                        "Living_Being":{
                            "Animal":{},
                            "Human":{},
                            "Plant":{}
                            }
                        }
                    ,
                "Concrete_Inanimate":{
                        "Material":{},
                        "Product":{
                            "Product":{
                                "Facility":{},
                                "Machine":{
                                    "Vehicle":{}
                                    }
                                }
                            }
                        }
                    }
                }   
            
}        

    


def _find(element, JSON, path='',root=0):    
    if element in JSON:
        path = path + element 
        return path 

    out_path = None

    
    for key in JSON:
        if isinstance(JSON[key], dict):
            tmp_path = _find(element, JSON[key],path + key+'/',root = root+1 )
            if tmp_path != None  : return tmp_path
            if root == 0: out_path = tmp_path 


    

def find_path(key,couche):
    all_path = _find(key,Arbre)
    if all_path == None :
        all_path = "Thing"
    split_path = all_path.split('/')
    if len(split_path)<= couche :
        return split_path[len(split_path)-1]
    else :
        return split_path[couche]

def couche(df,colonne,nb):
    df[colonne].apply(find_path,args=(nb,))
    return df


    


