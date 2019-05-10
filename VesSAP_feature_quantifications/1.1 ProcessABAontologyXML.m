%#########################################################################
% Parse the ontology.xml of the AllenBrainMouse_CCFv3_201710
%author Mihail Todorov 
xDoc = xmlread('D:\Atlases\AllenBrainMouse_CCFv3_201710\ontology.xml');

allListitems = xDoc.getElementsByTagName('structure');

parsedOntologyXML={'structureId', 'structureLR', 'structureAcronym', 'structureName', 'parent-structure-id', 'structurePath', 'structureColorHexTriplet'};

% replace id 997 with 0 because the labels of ABA use 0 for background, not 997
% this is also used for the structure path finding, so be careful with its
% modification!
specialID = '997'; 


elementPathDelimiter = '/';
for k = 0:allListitems.getLength-1
%for k = 2:2
   thisListitem = allListitems.item(k);
   
   structureId = char(thisListitem.getElementsByTagName('id').item(0).getFirstChild.getData);
   structureAcronym  = strrep(char(thisListitem.getElementsByTagName('acronym').item(0).getFirstChild.getData),'"','');
   structureName     = strrep(char(thisListitem.getElementsByTagName('name').item(0).getFirstChild.getData),'"','');
   structureParentId = char(thisListitem.getElementsByTagName('parent-structure-id').item(0).getFirstChild.getData);
   structureColorHexTriplet = char(thisListitem.getElementsByTagName('color-hex-triplet').item(0).getFirstChild.getData);
   
   if k>0
       structurePath = [structureParentId, elementPathDelimiter, structureId];
       structureParentIdID = structureParentId;
       while ~strcmp(structureParentIdID,specialID)
        thisListitem = thisListitem.getParentNode.getParentNode;
        structureParentIdID = char(thisListitem.getElementsByTagName('parent-structure-id').item(0).getFirstChild.getData);
        structurePath = [structureParentIdID, elementPathDelimiter, structurePath];
       end
   else
       structurePath = structureId;
   end
   
   
   if strcmp(structureId,specialID)
       structureId = '0';
       structureLR = 'LR';
   else
       structureLR = 'R';
   end 
   
   thisElementData = {structureId, structureLR, structureAcronym, structureName, structureParentId, structurePath, structureColorHexTriplet};
   parsedOntologyXML(end+1,:) = thisElementData;
   
   %create the negative ID and use L side
   if strcmp(structureId,'0')
       parsedOntologyXML(end,:) = thisElementData;
   else
       structureId = ['-', structureId];
       structureLR = 'L';
       thisElementData = {structureId, structureLR, structureAcronym, structureName, structureParentId, structurePath, structureColorHexTriplet};
       parsedOntologyXML(end+1,:) = thisElementData;
   end   
end

clear structureId structureLR structureAcronym structureName parent-structure-id structureColorHexTriplet structureParentId structureParentIdID structurePath
clear allListitems thisElement thisElementData thisList thisListitem xDoc
clear k
% 
% 
%#########################################################################
% create a new column based on the groups of structureColorHexTriplet
% NOTE: column posititions are hard coded, adjust if you change the 
% parsedOntologyXML is of different format!!
% NOTE: groups are not meant for storing L or R side of the structures,
% use it with the columns of structureLR if necessary!
parsedOntologyXML=[parsedOntologyXML repmat({''},size(parsedOntologyXML,1),1)];
uniquesOfstructureColorHexTriplet = unique(parsedOntologyXML(:,7),'stable');
for iHexUnique = 1:size(uniquesOfstructureColorHexTriplet,1)
    for iHexVal = 1:size(parsedOntologyXML,1)
        if strcmp(parsedOntologyXML{iHexVal,7},uniquesOfstructureColorHexTriplet{iHexUnique,1})
            parsedOntologyXML{iHexVal,8}=num2str(iHexUnique-2);
        end
    end   
end
parsedOntologyXML{1,8}='structureColorHexTripletGroup';
clear uniquesOfstructureColorHexTriplet iHexUnique iHexVal













