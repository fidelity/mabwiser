Search.setIndex({docnames:["about","api","contributing","examples","index","installation","new_bandit","quick"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["about.rst","api.rst","contributing.rst","examples.rst","index.rst","installation.rst","new_bandit.rst","quick.rst"],objects:{"mabwiser.base_mab":[[1,1,1,"","BaseMAB"]],"mabwiser.base_mab.BaseMAB":[[1,2,1,"","add_arm"],[1,3,1,"","arm_to_expectation"],[1,3,1,"","arm_to_status"],[1,3,1,"","arms"],[1,3,1,"","backend"],[1,4,1,"","cold_arms"],[1,2,1,"","fit"],[1,3,1,"","n_jobs"],[1,2,1,"","partial_fit"],[1,2,1,"","predict"],[1,2,1,"","predict_expectations"],[1,2,1,"","remove_arm"],[1,3,1,"","rng"],[1,4,1,"","trained_arms"],[1,2,1,"","warm_start"]],"mabwiser.mab":[[1,1,1,"","LearningPolicy"],[1,1,1,"","MAB"],[1,1,1,"","NeighborhoodPolicy"]],"mabwiser.mab.LearningPolicy":[[1,1,1,"","EpsilonGreedy"],[1,1,1,"","LinGreedy"],[1,1,1,"","LinTS"],[1,1,1,"","LinUCB"],[1,1,1,"","Popularity"],[1,1,1,"","Random"],[1,1,1,"","Softmax"],[1,1,1,"","ThompsonSampling"],[1,1,1,"","UCB1"]],"mabwiser.mab.LearningPolicy.EpsilonGreedy":[[1,3,1,"id0","epsilon"]],"mabwiser.mab.LearningPolicy.LinGreedy":[[1,3,1,"id1","epsilon"],[1,3,1,"id2","l2_lambda"],[1,3,1,"id3","scale"]],"mabwiser.mab.LearningPolicy.LinTS":[[1,3,1,"id4","alpha"],[1,3,1,"id5","l2_lambda"],[1,3,1,"id6","scale"]],"mabwiser.mab.LearningPolicy.LinUCB":[[1,3,1,"id7","alpha"],[1,3,1,"id8","l2_lambda"],[1,3,1,"id9","scale"]],"mabwiser.mab.LearningPolicy.Softmax":[[1,3,1,"id10","tau"]],"mabwiser.mab.LearningPolicy.ThompsonSampling":[[1,3,1,"id11","binarizer"]],"mabwiser.mab.LearningPolicy.UCB1":[[1,3,1,"id12","alpha"]],"mabwiser.mab.MAB":[[1,2,1,"","add_arm"],[1,3,1,"","arms"],[1,3,1,"","backend"],[1,4,1,"","cold_arms"],[1,2,1,"","fit"],[1,3,1,"","is_contextual"],[1,4,1,"id13","learning_policy"],[1,3,1,"","n_jobs"],[1,4,1,"id14","neighborhood_policy"],[1,2,1,"","partial_fit"],[1,2,1,"","predict"],[1,2,1,"","predict_expectations"],[1,2,1,"","remove_arm"],[1,3,1,"","seed"],[1,2,1,"","warm_start"]],"mabwiser.mab.NeighborhoodPolicy":[[1,1,1,"","Clusters"],[1,1,1,"","KNearest"],[1,1,1,"","LSHNearest"],[1,1,1,"","Radius"],[1,1,1,"","TreeBandit"]],"mabwiser.mab.NeighborhoodPolicy.Clusters":[[1,3,1,"id15","is_minibatch"],[1,3,1,"id16","n_clusters"]],"mabwiser.mab.NeighborhoodPolicy.KNearest":[[1,3,1,"id17","k"],[1,3,1,"id18","metric"]],"mabwiser.mab.NeighborhoodPolicy.LSHNearest":[[1,3,1,"id19","n_dimensions"],[1,3,1,"id20","n_tables"],[1,3,1,"id21","no_nhood_prob_of_arm"]],"mabwiser.mab.NeighborhoodPolicy.Radius":[[1,3,1,"id22","metric"],[1,3,1,"id23","no_nhood_prob_of_arm"],[1,3,1,"id24","radius"]],"mabwiser.mab.NeighborhoodPolicy.TreeBandit":[[1,3,1,"id27","tree_parameters"]],"mabwiser.simulator":[[1,1,1,"","Simulator"],[1,5,1,"","default_evaluator"]],"mabwiser.simulator.Simulator":[[1,3,1,"","arm_to_stats_test"],[1,3,1,"","arm_to_stats_total"],[1,3,1,"","arm_to_stats_train"],[1,3,1,"","arms"],[1,3,1,"","bandit_to_arm_to_stats_avg"],[1,3,1,"","bandit_to_arm_to_stats_max"],[1,3,1,"","bandit_to_arm_to_stats_min"],[1,3,1,"","bandit_to_arm_to_stats_neighborhoods"],[1,3,1,"","bandit_to_confusion_matrices"],[1,3,1,"","bandit_to_expectations"],[1,3,1,"","bandit_to_neighborhood_size"],[1,3,1,"","bandit_to_predictions"],[1,3,1,"","bandits"],[1,3,1,"","batch_size"],[1,3,1,"","contexts"],[1,3,1,"","decisions"],[1,3,1,"","evaluator"],[1,2,1,"","get_arm_stats"],[1,2,1,"","get_stats"],[1,3,1,"","is_ordered"],[1,3,1,"","is_quick"],[1,3,1,"","logger"],[1,2,1,"","plot"],[1,3,1,"","rewards"],[1,2,1,"","run"],[1,3,1,"","scaler"],[1,3,1,"","test_indices"],[1,3,1,"","test_size"]],"mabwiser.utils":[[1,6,1,"","Arm"],[1,1,1,"","Constants"],[1,6,1,"","Num"],[1,5,1,"","argmax"],[1,5,1,"","argmin"],[1,5,1,"","check_false"],[1,5,1,"","check_true"],[1,5,1,"","create_rng"],[1,5,1,"","reset"]],"mabwiser.utils.Constants":[[1,3,1,"","default_seed"],[1,3,1,"","distance_metrics"]],mabwiser:[[1,0,0,"-","base_mab"],[1,0,0,"-","mab"],[1,0,0,"-","simulator"],[1,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","property","Python property"],"5":["py","function","Python function"],"6":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:property","5":"py:function","6":"py:data"},terms:{"0":[1,3,6],"00129":4,"01":3,"05":1,"1":[0,1,3,4,5,7],"10":[1,3,4],"100":3,"1000":3,"100000":3,"11":[1,3],"1109":4,"111":3,"1142":4,"12":3,"123456":[1,3,6],"13":3,"15":[1,3],"17":[1,3,4,6,7],"18":3,"19":[3,4],"1d":3,"2":[0,1,3,4,5],"20":[1,3,4,6,7],"200":3,"2010":0,"2016":0,"2017":1,"2019":4,"2021":4,"21":3,"2150021":4,"22":3,"23":3,"25":[1,3,4,6,7],"26":3,"27":3,"2d":3,"3":[1,3,4,5],"30":[1,4],"31st":4,"33":3,"37":3,"38":3,"39":3,"4":[1,3,4],"42":[3,6],"48":3,"5":[1,3],"50":3,"52":3,"55":3,"56":3,"57":3,"6":[3,4,5],"65":3,"68":3,"7":[1,3],"75":3,"8":[2,3,4],"83":3,"9":[1,3,4,6,7],"909":4,"914":4,"99":3,"abstract":[1,6],"boolean":1,"break":6,"case":[1,3,6],"class":[1,6],"default":[1,2,6],"do":[0,1,2,6],"final":6,"float":[1,6],"function":[1,2,3,6],"import":[0,1,3,4,5,6,7],"int":[1,4,6],"long":0,"new":[0,1,2,3,4],"public":[2,4],"return":[1,6],"static":1,"super":6,"true":[1,3,6],"try":5,A:[0,1,3,4,6],As:[0,1],At:0,For:[0,1,5,6],If:[1,5,6],In:[0,1,2,3,6],It:[1,3,4,6],No:1,Not:1,OR:4,Or:6,That:6,The:[0,1,2,3,4,5,6],Then:1,There:[0,3,5],These:[1,6],To:[1,5,6],_:[3,6],__init__:[1,6],__version__:6,_arm_:1,_baserng:[1,6],_copy_arm:6,_fit_arm:6,_mycoolpolici:6,_onlin:1,_parallel_fit:6,_predict_context:6,_uptake_new_arm:[1,6],_valid:6,_validate_mab_arg:6,_warm_start:1,abl:[0,6],about:[1,3,4,6],abov:6,accept:1,access:[1,6],accident:6,accompani:2,accord:6,accordingli:1,action:[0,6],activ:6,actual:1,ad:[1,3,4],adam:1,add:[1,6],add_arm:[1,3,6],addit:[1,3,6],additional_layout:3,additional_revenu:3,adher:[2,4],adjust:1,advantag:6,advertis:[0,3],affect:2,after:[0,1],ag:3,agent:0,aggreg:1,al:0,algorithm:[0,1,4],alia:1,all:[1,2,5,6],allow:[3,6],alpha:[1,3,4,7],alreadi:1,also:[0,1],altern:5,among:[0,3],amount:3,an:[0,1,3,4,6,7],ani:[1,2,5,6],api:[2,4],append:3,appli:[1,6],applic:[0,2],approach:1,approxim:1,ar:[0,1,2,3,5,6],arang:3,argmax:1,argmin:1,argument:1,arm1:[1,4,6,7],arm2:[1,4,6,7],arm3:1,arm:[1,3,6,7],arm_to_expect:[1,6],arm_to_featur:[1,3,6],arm_to_predict:6,arm_to_stat:1,arm_to_stats_test:1,arm_to_stats_tot:1,arm_to_stats_train:1,arm_to_statu:1,arm_to_threshold:1,arms_to_stat:1,arms_to_stats_train:1,arrai:[1,3,6],articl:4,artif:4,artifici:4,arxiv:0,asctim:1,assert:[3,6],assertdictequ:6,assertequ:6,assertionerror:1,assess:6,assign:1,associ:1,assum:1,assumpt:1,attribut:6,author:4,automat:6,avail:[1,3,6],averag:[1,3],avg:[1,3],b:[3,4,6],back:6,backend:[1,6],bandit:[1,3],bandit_to_arm_to_stat:1,bandit_to_arm_to_stats_avg:[1,3],bandit_to_arm_to_stats_max:[1,3],bandit_to_arm_to_stats_min:[1,3],bandit_to_arm_to_stats_neighborhood:1,bandit_to_confusion_matric:1,bandit_to_expect:1,bandit_to_neighborhood_s:1,bandit_to_predict:1,base:[1,3,4,6,7],base_mab:4,basemab:[1,6],basetest:6,basic:[1,2],batch:[1,3],batch_siz:[1,3],bdist_wheel:5,becom:3,been:[1,6],befor:[1,6],behav:6,behavior:[1,6],behind:[3,6],being:1,below:3,bernard:4,best:[0,1,3,6],best_arm:6,beta:1,between:[0,1,4,6,7],binar:[1,6],binari:1,booktitl:4,bool:[1,6],both:4,bound1:1,bound:[1,3,4],branch:6,braycurti:1,build:5,built:[3,4],cach:5,calcul:[1,6],call:[0,1,6],callabl:[1,6],can:[0,1,3,4,5,6],canberra:1,cannot:[1,6],capabl:3,captur:1,carnegi:0,cd:5,cdist:1,center:4,certain:1,chang:6,changelog:6,chebyshev:1,check:[1,2,6],check_fals:1,check_tru:1,checkout:6,choic:3,choleski:1,choos:[0,1,7],chosen:1,chronolog:1,cite:4,cityblock:1,clean:2,click:3,click_rat:3,clinic:0,clone:5,closest:1,cluster:[1,4],code:[1,2,6],coeffici:1,coher:2,cold:[1,6],cold_arm:[1,6],cold_arm_to_warm_arm:6,collect:1,collis:1,column:1,com:5,commerc:3,common:1,commun:1,compar:1,comparison:4,compat:1,complet:[1,6],compon:4,computation:1,conceptu:3,concurr:1,conf:4,confer:4,confid:[1,3,4],confirm:5,confus:1,congratul:6,connect:6,conserv:1,consid:6,consist:[1,6],constant:1,constructor:[1,6],contain:[1,2],context:[0,1,4,6],contexts_test:3,contexts_train:3,contextu:[0,1],contextual_mab:1,continu:6,contribut:[4,6],control:1,convert:1,cool:6,copi:[1,6],core:[1,3,6],correct:1,correctli:6,correl:1,correspond:[1,3,6],cosin:1,count:1,counter:6,cours:0,covari:1,cover:6,cpu:1,creat:[1,6],create_rng:1,cumul:[0,1],custom:[1,3,5],d:1,danc:6,data:[1,3,4,6,7],datafram:[1,3],dataset:3,dblp:4,decid:[0,6],decis:[0,1,3,4,6,7],decisions_test:3,decisions_train:3,decisiontreeregressor:1,declar:[1,6],decomposit:1,decor:6,decreas:1,deem:1,deepcopi:6,def:[1,6],default_evalu:1,default_se:1,defin:[1,6],definit:[0,1],degre:1,denot:[1,6],depend:[1,3],descript:[1,6],deserv:6,design:[2,3],detail:6,determin:[1,6],determinist:[0,1],develop:6,dice:1,dict:[1,6],dictat:6,dictionari:[1,6],differ:[0,1,3,6],dimens:1,dimension:1,dir:5,directli:[1,6],directori:2,discov:5,displai:3,dist:5,distanc:[1,6],distance_metr:1,distance_quantil:[1,3,6],distribut:[0,1],divis:1,doc:6,docsrc:6,doctest:6,document:[2,6],doe:[1,6],doi:4,done:6,down:6,dure:[3,6],dxd:1,e:[1,3,4,6],each:[0,1,3,6],easiest:2,effect:[1,6],eg:1,elmachtoub:1,emerg:6,emili:4,empti:1,enabl:6,engin:0,ensur:[1,2],entir:1,entri:[1,6],epsilon:[1,3,4],epsilongreedi:[1,3],equal:3,error:6,estim:1,et:0,etc:1,euclidean:1,evalu:1,even:6,event:3,everi:[0,1,3,6],everyon:6,evolv:6,exampl:[1,4,5,6,7],excel:[0,4],except:1,exchang:1,execut:[1,6],exist:[0,1,2,6],exp:6,expect:[0,1,3,4,6,7],expens:1,experi:[1,3],explor:[1,3],expos:4,express:[1,6],extend:[1,6],face:0,factor:1,fals:[1,3,6],fashion:6,fast:[1,4],featur:[1,2,3,4,6],fidel:[4,5],field:[1,3,6],file:6,filter:1,first:[1,6],fit:[1,3,4,6,7],fit_arm:6,fit_transform:3,flag:[1,6],folder:[3,5,6],follow:[1,4,6],foremost:6,format:1,formula:6,frac:1,frame:[1,3],framework:6,free:[1,4],from:[0,1,3,4,5,6,7],fromkei:6,g:[1,6],gener:[1,2,3,6],get:[1,2],get_arm_stat:1,get_stat:1,git:[5,6],github:[2,4,5,6],give:1,given:[0,1,3,6],global:1,go:6,goe:[1,6],good:1,greater:1,greedi:[1,3,4,6],greedili:1,guarante:1,guidelin:6,ha:[0,1,6],ham:1,handl:6,happi:2,hash:1,have:[0,1,2,6],heavili:4,helper:1,here:[3,6],high:3,higher:1,highest:[0,1],hint:6,histor:[1,3],histori:[1,3],homepag:3,host:[2,4],how:[1,3,4,5,6,7],http:[4,5],hyper:[1,3,4,6],hyper_parameter_tun:3,hyperplan:1,i:[1,3],i_d:1,ictai:4,idea:[3,6],ident:1,ieee:4,ijait:4,imagin:6,immut:6,implement:[1,4],implementor:[1,6],includ:[5,6],incorpor:1,increas:1,increment:6,independ:3,index:[1,4],indic:1,induc:1,infin:1,inform:[0,1,3],inherit:[1,6],initi:[1,6],inner:6,inproceed:4,input:[1,6],instal:[4,6],instead:[1,6],integ:1,intel:4,intellig:4,intend:1,interfac:[1,6],intern:[1,4,6],internet:0,interpret:1,introduc:[1,6],invers:1,invest:4,involv:6,is_contextu:1,is_minibatch:1,is_ord:[1,3],is_per_arm:[1,3],is_predict:6,is_quick:1,is_train:1,is_warm:1,isn:6,issu:[2,4],item:[1,6],its:[1,6],j:4,jaccard:1,joblib:1,journal:4,k:[0,1,4],kadioglu:4,keep:1,kei:1,keyword:6,kleynhan:4,kmean:1,knearest:1,know:0,kulsinski:1,kwarg:1,l2_lambda:[1,3],lambda:1,larg:3,later:6,latest:5,layout:3,lead:3,leaf:1,learn:[0,1,3,4,6,7],learning_polici:[1,3,6],learningpolici:[1,3,4,5,6,7],learnt:3,least:1,leav:1,length:1,less:1,let:6,level:3,levelnam:1,leverag:6,li:0,librari:[1,4,6,7],like:6,likelihood:1,linear:3,lingreedi:[1,4],lint:[1,4],linucb:[1,3,4],list:[1,3,5,6],list_of_arm:1,listen:3,live:6,local:[1,6],lock:1,log:1,log_fil:1,log_format:1,logger:1,logist:1,loki:1,look:6,lot:1,low:1,lp:6,lsh:4,lshnearest:1,lu:0,m:5,mab1:1,mab2:1,mab:[0,4,5,6,7],mab_nam:3,mabwis:[2,3,5,6,7],machin:0,made:1,magic:6,mahalanobi:1,mai:0,main:[2,6],make:[0,1,2,3,4,6],make_classif:3,mani:[0,1],map:[1,6],marek:1,master:[5,6],match:1,math:6,matric:1,matrix:1,max:1,maxim:0,maximum:1,mcnelli:1,md:6,mean:[1,3],mellon:0,member:6,memori:1,merg:6,messag:1,meta:6,method:[1,2,3,6],metric:[1,3],might:6,min:1,minibatchkmean:1,minimum:[1,6],minkowski:1,mismatch:1,miss:1,mode:6,model:[0,1,3,4,6,7],model_select:3,model_to_confusion_matric:1,model_to_predict:1,models_to_evalu:1,modif:2,modifi:1,modul:[1,4],more:[0,1],most:6,move:6,mu:1,mu_i:1,much:0,multi:1,multipl:[0,1,3],multipli:1,multiprocess:1,multivari:1,music:3,must:1,my_paramet:6,my_results_:3,my_value_to_arm:6,mycoolbandittest:6,mycoolpolici:6,n:[1,3],n_cluster:1,n_context_col:1,n_dimens:1,n_featur:3,n_i:1,n_inform:3,n_job:[1,3,6],n_row:1,n_sampl:3,n_tabl:1,name:[1,6],namedtupl:6,nan:1,nd:1,ndarrai:[1,6],nearest:[1,4],necessari:[5,6],need:[0,3,6],neg:[1,6],neighbor:1,neighborhood:[1,3,4,6],neighborhood_polici:[1,3],neighborhoodpolici:[1,3,4,5,6,7],next:[3,6],nn:1,no_nhood_prob_of_arm:1,non:[1,4],none:[1,3,5,6],noreturn:[1,6],normal:1,note:[1,6],notebook:5,notic:[1,6],notimplementederror:1,novemb:4,now:6,np:[1,3,6],nparrai:1,num:[1,6],num_run:6,number:[1,4,6],numer:1,numpi:[1,3,6],numpydoc:2,o:1,object:[0,1,6],observ:[0,1,3,6],obtain:1,offlin:[1,3],offline_sim:1,oh:1,older:2,one:[1,6],onli:[0,1],onlin:[0,1,3,6],onto:[1,6],oper:[1,6],optim:3,option:[0,1,3,6],order:[1,3],org:4,origin:[1,5],other:[0,1,3,6],otherwis:1,our:[2,6],out:[1,2],outcom:0,output:1,over:1,overal:0,overhead:1,overrid:1,own:6,p:1,packag:[5,6],page:4,pairwis:6,panda:[1,3],parallel:[1,4,6],paralleliz:4,param:6,paramet:[1,3,4,6],parametr:[1,4],part:[0,6],partial:1,partial_fit:[1,3,6],partit:1,pass:[2,5,6],past:3,pd:[1,3],pep:[2,4],per:3,per_arm:1,perform:[1,6],petrik:1,phase:[3,6],pip:[5,6],pipelin:1,plai:0,platform:5,playlist:3,pleas:[4,6],plot:[1,3],point:[1,6],polici:[1,3,6,7],pool:1,popular:[1,4,6],portland:4,posit:1,possibl:[3,6],potenti:[0,1],practic:1,precis:1,predict:[1,3,4,6,7],predict_expect:[1,3,6],prefix:[3,6],prepar:6,preprocess:[1,3],present:1,previou:[1,2,3,6],print:3,privat:6,probabl:1,probe:3,problem:[0,1,3],proc:0,procedur:1,process:1,progress:1,project:[1,2,5],promot:1,properli:6,properti:[1,6],proportion:1,prototyp:[1,4],provid:[1,4,6],publish:4,pull:[0,4,5],purpos:1,py3:5,py:[5,6],pydoc:6,python:[1,4,5,6],quantil:1,question:0,quickli:6,radiu:[1,3,4],rais:[1,6],randint:3,random:[1,3,4,6],random_st:3,randomli:1,rang:3,rate:3,rather:1,ration:1,re:3,read:1,readi:6,readm:6,real:[0,6],realiti:3,receiv:[0,1],recommend:[0,1,3],recompil:6,refer:[0,5],reflect:1,region:1,regist:6,regress:1,regular:1,releas:4,relev:1,reli:1,remain:6,remov:[1,6],remove_arm:1,renew:0,repeat:1,repo:[2,3,5],represent:1,request:4,requir:[1,2,4],research:[0,1,4],reset:[1,6],resourc:0,result:[0,1,3],retain:1,retriev:[1,3],revenu:3,review:6,reward:[0,1,3,4,6,7],rewards_test:3,rewards_train:3,ridg:1,rng:[1,6],robust:1,rogerstanimoto:1,root:1,routin:1,row:1,rst:6,run:[0,1,3,5,6],russellrao:1,ryan:1,s0218213021500214:4,s:[1,3,4,6],same:[1,6],sampl:[0,1,2,4],save:3,save_result:3,scale:[1,3],scaler:[1,3,6],scenario:3,scikit:4,scipi:1,scratch:[5,6],sdist:5,search:6,sechan:1,section:[2,6],see:[1,6],seed:[1,3,6],seen:6,select:1,self:6,send:4,sensit:1,separ:1,sequenc:0,sequenti:0,serdar:4,seri:[1,3,6],servic:3,set:[0,1,3,6],setup:4,setuptool:5,seuclidean:1,share:[1,6],shell:5,should:[1,2,3,5],show:[3,4,7],shown:[3,6],side:0,sigma:1,sign:1,signatur:[1,6],sim:3,simhash:1,similar:[1,6],simpl:[1,6],simul:4,sinc:3,singl:1,situat:0,size:[1,3],skeleton:1,skip:[1,3],sklearn:[1,3],small:6,so:[1,6],softmax:[1,4],sokalmichen:1,sokalsneath:1,solv:[1,3],some:[0,1,6],someth:0,space:1,spatial:1,specif:1,specifi:1,speed:1,sphinx:6,split:[1,3],sqeuclidean:1,sqrt:1,squar:1,standard:[2,3,4],standardscal:[1,3],start:[1,2,3,6],start_index:[1,6],stat:[1,3],statis:3,statist:1,statu:1,std:1,step:[1,3,6],stick:0,still:2,stochast:0,store:[1,6],str:[1,3,6],stream:3,strength:1,strengthen:6,string:[1,6],strong:4,strongkk19:4,strongkk21:4,style:4,sub:1,subscrib:3,success:[1,5,6],suffer:1,suit:6,suitabl:0,sum:[1,6],support:[1,3,4],sure:[2,6],survei:0,system:6,t:[0,1,6],tabl:1,take:[1,6],tau:1,tell:6,temperatur:1,test:[0,1,2,3,4,7],test_:6,test_add_new_arm:6,test_bas:6,test_df:3,test_df_revenu:3,test_fit_twic:6,test_indic:1,test_input_typ:6,test_invalid:6,test_my_paramet:6,test_mycoolbandit:6,test_parallel:6,test_partial_fit:6,test_remove_existing_arm:6,test_simple_usecase_arm:6,test_simple_usecase_expect:6,test_siz:[1,3],test_unused_arm:6,test_warm_start:6,test_within_neighborhood_polici:6,test_zero_reward:6,than:1,thei:[0,1,6],them:1,therefor:1,thi:[0,1,3,6],thompson:[1,4],thompsonsampl:1,those:6,thread:1,three:6,time:[0,1,6],titl:4,todo:6,togeth:1,toi:3,too:6,tool:4,total:[1,3,6],track:[1,4],train:[1,3,4,6,7],train_df:3,train_test_split:[1,3],trained_arm:1,transform:3,tree:1,tree_paramet:1,treebandit:[1,4],treeheurist:1,trial:[0,1],ts:4,tune:[1,3,4],tupl:1,two:[4,5,6,7],txt:[5,6],tyler:0,type:[1,6],typeerror:1,typic:6,uai:1,ucb1:[1,3,4,7],ucb:1,uncertainti:0,under:[0,1,6],underli:[0,6],uniform:1,uniformli:1,uninstal:6,union:1,uniqu:1,unit:[1,2,6],unittest:[5,6],univers:0,unknown:0,unlik:6,untrain:[1,6],unus:6,up:1,updat:[1,3,6],upgrad:4,upper:[1,3,4],url:4,us:[1,2,3,4,5,6,7],usa:4,usag:[4,5,6],user:[3,6],util:[4,6],valid:[1,6],valu:[1,6],valueerror:1,varianc:1,variant:0,vector:1,veri:1,version:[1,5,6],via:6,victori:6,violat:1,volum:4,wa:[1,5],wai:2,want:[2,3],warm:[1,3,6],warm_arm:6,warm_start:[1,3,6],warm_started_bi:1,we:[0,2,3,6],websit:3,weight:[1,3],were:[1,6],what:[0,3,6],wheel:5,when:[1,6],where:[0,1,6],whether:[1,3],which:[0,1,3,6],whl:5,within:[1,6],work:[1,2,6],worker:1,world:[0,6],worst:3,would:6,wrapper:6,x:[1,5],x_i:1,year:4,yet:0,you:[1,2,3,4,5,6],your:[4,6],zero:[1,6],zhou:0},titles:["About Multi-Armed Bandits","MABWiser Public API","Contributing","Usage Examples","MABWiser Contextual Multi-Armed Bandits","Installation","Adding a New Bandit","Quick Start"],titleterms:{"1":6,"2":6,"3":6,"4":6,"new":6,"public":[1,6],about:0,ad:6,algorithm:6,api:[1,6],arm:[0,4],avail:4,bandit:[0,4,6],base_mab:1,bug:4,citat:4,code:[4,5],content:4,context:3,contextu:[3,4],contribut:2,exampl:3,exploit:0,explor:0,expos:6,free:3,high:6,implement:6,indic:4,instal:5,level:6,librari:5,mab:[1,3],mabwis:[1,4],multi:[0,4],non:3,option:5,overview:6,parallel:3,parametr:3,polici:4,pull:6,quick:[4,7],report:4,request:6,requir:5,send:6,setup:5,simul:[1,3],sourc:[4,5],start:[4,7],tabl:4,test:[5,6],upgrad:5,usag:3,util:1,vs:0,your:5}})