%% Split NIFTI volumes by major region boundaries
% be aware that neighboring labels will be included as this is just copying a bounding box
% this will include some duplication of labels as well as bstem, and hc
% will have labels from the ctx; cereb will have some labels from bstem
% etc..

vol = niftiread("T:\CD1_42\elastix49_out30\transformix_out\result.nii");
outputFolder  = "T:\CD1_42\elastix49_out30\transformix_out\"; 

%less memory intensive
filenameOutAbsoluteHC = 'BL6n2-iso3um-bin081-hcBoundningTotalMasked.nii';
filenameOutAbsoluteCEREB = 'BL6n5-iso3um-bin081-cerebBoundningTotalMasked.nii';
filenameOutAbsoluteBSTEM = 'BL6n5-iso3um-bin081-bstemBoundningTotalMasked.nii';
filenameOutAbsoluteCTX = 'CD1-42-iso3um-bin081-ctxBoundningTotalMasked.nii';


%less memory intensive   
niftiwrite(vol(320:2488,1344:2362,431:1728),[outputFolder+filenameOutAbsoluteHC],'Combined',false);
niftiwrite(vol(310:2839,3028:4084,198:1678),[outputFolder+filenameOutAbsoluteCEREB],'Combined',false);
niftiwrite(vol(670:2464,1439:4119,273:2036),[outputFolder+filenameOutAbsoluteBSTEM],'Combined',false);
niftiwrite(vol(239:2877,192:3116,104:2085),[outputFolder+filenameOutAbsoluteCTX],'Combined',false);

%% Map centerlines and bifurcations to atlas regions
% this section is used for centerline and bifurcation quantification
% multiply labels by features of intensity 1 to include only voxels of desired major region

volLabels = niftiread("T:\CD1_41\elastix49_out30\transformix_out\hcBoundingBox.hdr");
volVoxels = niftiread("T:\CD1_41\elastix49_out20\pp_skeleton_CD1-41-iso3um-segm-bin081-hcTotalMasked.nii");
volLabels=volLabels(1:size(volVoxels,1),1:size(volVoxels,2),1:size(volVoxels,3));

volVoxels = single(volVoxels);

vol = volLabels .* volVoxels;

uniqueValsDirty = zeros(1,2);
for i=1:size(vol,3)
    volSlice=vol(:,:,i);
    [C,ia,ic] = unique(volSlice);
    a_counts = accumarray(ic,1);
    value_counts = [C, a_counts];
    uniqueValsDirty = cat(1,uniqueValsDirty,value_counts);
    clear C ia ic a_counts value_counts volSlice
end


UniquesOfuniqueValsDirty = unique(uniqueValsDirty(:,1));
uniqueVals = zeros(1,2);
for i=1:size(UniquesOfuniqueValsDirty,1)
    tmpi = find(uniqueValsDirty(:,1)==UniquesOfuniqueValsDirty(i));
    uniqueVals(i,1) = UniquesOfuniqueValsDirty(i);
    uniqueVals(i,2) = sum(uniqueValsDirty(tmpi,2));
    clear tmpi
end
clear i UniquesOfuniqueValsDirty uniqueValsDirty

%% Map radii to atlas regions
volLabels = niftiread("/mnt/C286054386053985/mtodorov/nifti-files/BL6n4/elastix49_out30/transformix_out/BL6n2-iso3um-bin081-hcBoundningTotalMasked.hdr");
volVoxels = niftiread("/mnt/C286054386053985/mtodorov/nifti-files/BL6n4/elastix49_out20/pp_skeleton_BL6n2-iso3um-segm-bin081-hcTotalMasked.nii");
volLabels=volLabels(1:size(volVoxels,1),1:size(volVoxels,2),1:size(volVoxels,3));


volVoxels = single(volVoxels);
volMasked = volLabels .* volVoxels;
clear volLabels volVoxels

volRadius = niftiread("/mnt/C286054386053985/mtodorov/nifti-files/BL6n4/elastix49_out20/pp_radius_BL6n2-iso3um-segm-bin081-hcTotalMasked.nii");
%volRadius=int8(volRadius);


stats_table = zeros(1239,2);
uniqueVals_CD142_v3 = uniqueVals_BL6n2_v3_hcSegmPPskel; %uniqueVals_CD141_v3_ctxSegmPPskel
for i=1:size(uniqueVals_CD142_v3(:,1),1)
    stats_table(i,1) = uniqueVals_CD142_v3(i,1);
    
    % is already extracted
	%V_ind = volLabels == uniqueVals_CD142_v3(i,1);
	%V_ind = int8(V_ind);                              
    %stats_table(i,2) = sum(V_ind(:));
    
    A_ind = volMasked == uniqueVals_CD142_v3(i,1);
    A_ind = int8(A_ind);

    % is already extracted
    %stats_table(i,3) = sum(A_ind(:));

    M = (A_ind .* volRadius);
    
    mean_rad = mean(M(M~=0));
    disp(['at label ' , num2str(stats_table(i,1)) , ' mean radius is ' , num2str(mean_rad)])

    stats_table(i,2) = mean_rad;
end
disp('Script finished.')
clear volMasked A_ind M i mean_rad


%% Sort features into the global sheet "parsedOntologyXML"
% one can use the supplied parsedOntologyXML.mat - used for manuscript.mat

%adjust these
variableArray = uniqueVals_CD142_v3_ctxSegmPPradius;
coulmnNumber = 38; %size(parsedOntologyXML,2)+1; %coulmnNumber = 9;
columnHeader = 'CD142_radius';

%dont touch
staticArray = zeros(size(parsedOntologyXML,1),1);
staticArray(1) = NaN;
for i=2:size(parsedOntologyXML,1)
    staticArray(i)= str2double(parsedOntologyXML{i,1});
end
for i=1:size(variableArray,1)
    k = find(staticArray==variableArray(i,1));
    if ~isempty(k)
        parsedOntologyXML{k,coulmnNumber} = num2str(variableArray(i,2));
    end
end
parsedOntologyXML{1,coulmnNumber} = columnHeader;
clear variableArray


%% Create secondary, aggregated list of higher order groups
%  NOTE: RENAME AFTER SUCCESSFUL CREATION!
% parsedOntologyXML(:,9)  .. BL6n2_v3_region .. groupedValues(:,2)
% parsedOntologyXML(:,10) .. BL6n4_v3_region .. groupedValues(:,3)
% parsedOntologyXML(:,11) .. BL6n5_v3_region .. groupedValues(:,4)

% parsedOntologyXML(:,12) .. CD115_v3_region .. groupedValues(:,5)
% parsedOntologyXML(:,13) .. CD141_v3_region .. groupedValues(:,6)
% parsedOntologyXML(:,14) .. CD141_v3_region .. groupedValues(:,7)

% parsedOntologyXML(:,21) .. BL6n2_v3_skeleton .. groupedValues(:,8)
% parsedOntologyXML(:,22) .. BL6n4_v3_skeleton .. groupedValues(:,9)
% parsedOntologyXML(:,23) .. BL6n5_v3_skeleton .. groupedValues(:,10)

% parsedOntologyXML(:,24) .. CD115_v3_skeleton .. groupedValues(:,11)
% parsedOntologyXML(:,25) .. CD141_v3_skeleton .. groupedValues(:,12)
% parsedOntologyXML(:,26) .. CD142_v3_skeleton .. groupedValues(:,13)

% parsedOntologyXML(:,27) .. BL6n2_v3_bifurcat .. groupedValues(:,14)
% parsedOntologyXML(:,28) .. BL6n4_v3_bifurcat .. groupedValues(:,15)
% parsedOntologyXML(:,29) .. BL6n5_v3_bifurcat .. groupedValues(:,16)

% parsedOntologyXML(:,30) .. CD115_v3_bifurcat .. groupedValues(:,17)
% parsedOntologyXML(:,31) .. CD141_v3_bifurcat .. groupedValues(:,18)
% parsedOntologyXML(:,32) .. CD142_v3_bifurcat .. groupedValues(:,19)

% parsedOntologyXML(:,33) .. BL6n2_v3_radius   .. groupedValues(:,20)
% parsedOntologyXML(:,34) .. BL6n4_v3_radius   .. groupedValues(:,21)
% parsedOntologyXML(:,35) .. BL6n5_v3_radius   .. groupedValues(:,22)

% parsedOntologyXML(:,36) .. CD115_v3_radius   .. groupedValues(:,23)
% parsedOntologyXML(:,37) .. CD141_v3_radius   .. groupedValues(:,24)
% parsedOntologyXML(:,38) .. CD142_v3_radius   .. groupedValues(:,25)

% groupedValues(:,2) .. BL6n2_v3_region
% groupedValues(:,3) .. BL6n4_v3_region
% groupedValues(:,4) .. BL6n5_v3_region

% groupedValues(:,6) .. BL6_mean
% groupedValues(:,7) .. BL6_sd
% groupedValues(:,8) .. BL6_densityOfSkeletonPerArea
% groupedValues(:,9) .. BL6_densityOfBifurcationsPerArea

columnOfparsedOntologyXMLToBeExtracted = 38;
columnOfgroupParamsUniquesCountsToBeInserted = 7; %1st column are the group IDs
calculateMeanInstedSum = true;

groupParams = NaN(size(parsedOntologyXML,1)-1,1);
for i=2:size(parsedOntologyXML,1)
    groupParams(i) = str2num(parsedOntologyXML{i,8});
end


[C,ia,ic] = unique(groupParams);
a_counts = accumarray(ic,1);
groupParamsUniquesCounts = [C, a_counts];
clear i C ia ic a_counts

for i=1:size(groupParamsUniquesCounts,1)
%for i=1:8
    idx = find(groupParamsUniquesCounts(i)==groupParams);
    counts=[];
    for ii=1:size(idx,1)        
        if ~isempty(parsedOntologyXML{idx(ii),columnOfparsedOntologyXMLToBeExtracted})
            counts(end+1) = str2num(parsedOntologyXML{idx(ii),columnOfparsedOntologyXMLToBeExtracted});
        end
    end
    groupedValues(i,1) = groupParamsUniquesCounts(i,1);
    if calculateMeanInstedSum
        countV=nanmean(counts);
    else
        countV=sum(counts);
    end
    groupedValues(i,columnOfgroupParamsUniquesCountsToBeInserted) = countV;
    disp(['id ',num2str(groupedValues(i,1)),'   countV ',num2str(countV)])
end

groupColorHexTriplets={size(groupedValues,1),2};
for i=1:size(groupedValues,1)
    groupid1 = num2str(groupedValues(i,1));
    for ii=1:size(parsedOntologyXML,1)
        groupid2=parsedOntologyXML{ii,8};
        if strcmp(groupid1, groupid2)
            groupColorHexTriplets{i,1} = groupid1;
            groupColorHexTriplets{i,2} = parsedOntologyXML{ii,7};
        end
    end
end
clear groupid1 groupid2
clear i ii columnToBeExtracted groupParamsUniquesCounts counts countV idx groupParams columnOfparsedOntologyXMLToBeExtracted columnOfgroupParamsUniquesCountsToBeInserted

%% END of script
