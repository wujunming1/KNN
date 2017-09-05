from math import log2
import numpy as np
import pickle
def calcShannongent(dataset):#计算信息熵，所有类别所有可能值包含的信息期望值
    m=len(dataset)#数据实例的总数
    labelcount={}
    for d in dataset:
        label=d[-1]
        labelcount[label]=labelcount.get(label,0)+1
    shannonEnt=0.0
    for key in labelcount:
        probability=labelcount[key]/m
        shannonEnt-=probability*log2(probability)
    return shannonEnt
def createDataset():
    dataset=[[1,1,"yes"],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=["no surfacing","flippers"]
    return dataset,labels
def calcnum():#统计每个数字出现的次数
    list1=[2,4,5,4,6,6,7,3,4,3,3,3]
    listcount={}
    for i in list1:
        listcount[i]=listcount.get(i,0)+1
    for key,value in listcount.items():
        print(key,value)
def splitDataset(dataset,axis,value):#dataset表示带划分的数据集,axis表示划分数据集的特征,需要返回的特征的值
    retDataset=[]
    for featVec1 in dataset:
        if featVec1[axis]==value:
            reducedFeatVec=featVec1[:axis]
            reducedFeatVec.extend(featVec1[axis+1:])
            retDataset.append(reducedFeatVec)
    return retDataset
def chooseBestFeatureToSplit(dataset):
    numFeatures=len(dataset[0])-1
    baseEntropy=calcShannongent(dataset)
    bestinfoGain=0.0;bestfeature=-1
    for i in range(numFeatures):
        featlist=[example[i] for example in dataset]
        uniquefeatVal=set(featlist)
        newEntropy=0.0
        for value in uniquefeatVal:
            subDataset=splitDataset(dataset,i,value)
            probablity=len(subDataset)/float(len(dataset))
            newEntropy+=probablity*calcShannongent(subDataset)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestinfoGain):
            bestinfoGain=infoGain
            bestfeature=i
    return bestfeature
def majorityCnt(classlist):
    classcount={}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote]=0
        classcount[vote]+=1
    sortedclasscount=sorted(classcount.items(),key=lambda item:item[1],reverse=True)
    return sortedclasscount[0][0]
def createTree(dataset,labels):
    classlist=[example[-1] for example in dataset]
    if classlist.count(classlist[0])==len(classlist):
        return classlist[0]#如果类别完全相同则停止继续划分(第一种情况）
    if len(dataset[0])==1:#（第二种情况）,如果遍历完所有特征是返回出现次数最多的分类，即投票法
        return majorityCnt(classlist)
    bestfeat=chooseBestFeatureToSplit(dataset)
    bestfeatlabel=labels[bestfeat]
    myTree={bestfeatlabel:{}}
    del(labels[bestfeat])
    featvalues=[example[bestfeat] for example in dataset]
    uniqueVals=set(featvalues)
    for value in uniqueVals:
        sublabels=labels[:]
        myTree[bestfeatlabel][value]=createTree(splitDataset(dataset,bestfeat,value),sublabels)
    return myTree
def classify(inputtree,featLabels,testVec):
    firstindice=list(inputtree.keys())
    firstStr=firstindice[0]
    sencondDict=inputtree[firstStr]
    featindex=featLabels.index(firstStr)
    for key in sencondDict.keys():
        if testVec[featindex]==key:
            if type(sencondDict[key]).__name__=='dict':
                classlabel=classify(sencondDict[key],featLabels,testVec)
            else:
                classlabel=sencondDict[key]
    return classlabel
def getNumLeafs(myTree):
    #获取决策树叶子节点的数目
    numLeafs=0
    firstindice=list(myTree.keys())
    firstStr=firstindice[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=="dict":
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs
def getTreeDepth(myTree):#获得树的最大深度
    maxdepth=0
    firstindice=list(myTree.keys())
    firstStr=firstindice[0]
    SencondDict=myTree[firstStr]
    for key in SencondDict.keys():
        if type(SencondDict[key]).__name__=="dict":
            thisdepth=1+getTreeDepth(SencondDict[key])
        else:
            thisdepth=1
        if thisdepth>maxdepth:maxdepth=thisdepth
    return maxdepth
def storeTree(myTree,filename):
    fw=open(filename,"wb")
    pickle.dump(myTree,fw,1)
    fw.close()
def grabTree(filename):
    fr=open(filename,"rb")
    return pickle.load(fr)
if __name__=="__main__":
    dataset,labels=createDataset()
    myTree=createTree(dataset,labels)
    storeTree(myTree,"1.txt")
    outputTree=grabTree("1.txt")
    print(outputTree)
    print(getNumLeafs(myTree))
    print(getTreeDepth(myTree))
    # firstindice=list(myTree.keys())
    # print(firstindice[0])
    # dist1={1:2,"wujunming":123,"23":12}#这是因为python3改变了dict.keys,返回的是dict_keys对象,
    # # 支持iterable 但不支持indexable，我们可以将其明确的转化成list
    # print(dist1.keys())
    # print(list(dist1.items()))
    # for key in dist1:
    #     print(dist1.get(key))
    # print(myTree)
    labels=["no surfacing","flippers"]
    classlabel=classify(myTree,labels,[1,1])
    print(classlabel)
    # str="ret vs fsgrsgsg gdgvs fs vs"
    # print(str.count("vs",0,12))
    # x=np.array([1,2,3])
    # x=list(x)
    # print(x)
    # x=[1,2,3,4]
    # del x[1:]#删除指定索引位置的元素
    # print(x)
    # y=[1,3,4,5,3]
    # y.remove(3)
    # print(y)
    # c=y.pop(1)#此时pop函数传递的参数为元素在列表中的索引位置
    # print(c)
    # print(y)
    elementlist=[7,1,2,2,3,4,5,5,6,5,5,7]
    elementset=set(elementlist)
    countmax=0
    for value in elementset:
        count=elementlist.count(value)
        if(count>countmax):
            countmax=count
            value1=value
    print("the number of %d is %d"%(value1,countmax))
    # print(chooseBestFeatureToSplit(dataset))
    # # dataset[0][-1]="maybe"
    # print(dataset)
    # shanonent=calcShannongent(dataset)
    # print(shanonent)
    # calcnum()
    # retdataset=splitDataset(dataset,1,1)
    # print(retdataset)
    fr=open("lenses.txt")
    dataset1=[line.strip().split('\t')for line in fr.readlines()]
    lenseslabels=['age','prescript','astigmatic','tearRate']
    lensesTree=createTree(dataset1,lenseslabels)
    print(lensesTree)
    print(lenseslabels)




