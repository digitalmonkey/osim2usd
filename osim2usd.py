# Library imports
import math, os, sys, getopt

# For XML parsing
import xml.etree.ElementTree as xmlTree

# For OpenSim support
import opensim as osim

# For USD support
from pxr import Sdf, Gf, Usd, UsdGeom, UsdSkel, Vt

# For vtk support (reading vtp geometry)
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def getMeshArrays(meshPath):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(meshPath)
    reader.Update()
    polyDataOutput = reader.GetOutput()

    faces = polyDataOutput.GetPolys()
    cellCount = faces.GetNumberOfCells()
    faceArray = vtk_to_numpy(faces.GetData())

    i = 0
    faceVertexCounts = []
    faceVertexIndices = []
    for polyIndex in range(cellCount):
        faceVertexCounts.append(faceArray[i].item())
        for v in range(faceArray[i]):
            faceVertexIndices.append(faceArray[i + 1 + v].item())
        i += (faceArray[i] + 1)
    assert( cellCount == len(faceVertexCounts))


    points = polyDataOutput.GetPoints()
    points = vtk_to_numpy(points.GetData())

    return (faceVertexCounts, faceVertexIndices, points)

def writeUsd(parseTree, usdPath, geomPath):
    root = parseTree.getroot()
    stage = Usd.Stage.CreateNew(usdPath)

    for model in root.findall("./Model"):
        print("Model: ", model.attrib["name"])
        skelRootPath = "/" + model.attrib["name"]
        modelPrim = UsdSkel.Root.Define(stage, skelRootPath)
        stage.SetDefaultPrim(stage.GetPrimAtPath(skelRootPath))

        meshBodyDict = dict()
        for bodyset in model.findall("./BodySet"):
            print("Bodyset: ", bodyset.attrib["name"])
            skeletonPath = skelRootPath + "/" + bodyset.attrib["name"]
            skeleton: UsdSkel = UsdSkel.Skeleton.Define(stage, skeletonPath)

            for body in bodyset.findall("./objects/Body"):
                print("\tBody: ", body.attrib["name"])
                # bodyPrim = UsdGeom.Xform.Define(stage,"/" + model.attrib["name"] + "/" + bodyset.attrib["name"] + "/" + body.attrib["name"])

                for mesh in body.findall("./attached_geometry/Mesh"):
                    print("\t\tMesh: ", mesh.attrib["name"])

                    meshPath = skelRootPath + "/" + mesh.attrib["name"]
                    meshGeom = UsdGeom.Mesh.Define(stage, meshPath)
                    meshPrim = stage.GetPrimAtPath(meshPath)
                    meshBodyDict[meshGeom] = body.attrib["name"]

                    scaleFactor = mesh.find("./scale_factors")
                    scaleFactors = [float(x) for x in scaleFactor.text.split()]
                    meshScaleOp = meshGeom.AddScaleOp()
                    meshScaleOp.Set(Gf.Vec3f(scaleFactors))

                    color = mesh.find("./Appearance/color")
                    # for color in mesh.findall("./Appearance/color"):
                    colors = [float(x) for x in color.text.split()]
                    colorAttr = meshGeom.GetDisplayColorAttr()
                    colorAttr.Set([tuple(colors)])

                    opacity = mesh.find("./Appearance/opacity")
                    opacityValue = float(opacity.text)
                    opacityAttr = meshGeom.GetDisplayOpacityAttr()
                    opacityAttr.Set([opacityValue])

                    meshFile = mesh.find("./mesh_file")
                    print("\t\t\tMeshFile: ", meshFile.text)

                    meshPath = geomPath + "/" + meshFile.text
                    ( faceVertexCounts, faceVertexIndices, points ) = getMeshArrays(meshPath)

                    faceVertexCountsAttr = meshPrim.CreateAttribute('faceVertexCounts', Sdf.ValueTypeNames.IntArray)
                    faceVertexCountsAttr.Set(Vt.IntArray(faceVertexCounts))

                    faceVertexIndicesAttr = meshPrim.CreateAttribute('faceVertexIndices', Sdf.ValueTypeNames.IntArray)
                    faceVertexIndicesAttr.Set(Vt.IntArray(faceVertexIndices))

                    pointsAttr = meshPrim.CreateAttribute('points', Sdf.ValueTypeNames.Float3Array)
                    pointsAttr.Set(Vt.Vec3fArray.FromNumpy(points))

            for wrappedObject in body.findall("./WrapObjectSet/objects/*"):
                print("\t\t WrappedObject[", wrappedObject.tag, "] = ", wrappedObject.attrib["name"])

        print("meshBodyDict: ", meshBodyDict)

        # Parse joints
        jointNames = Vt.TokenArray([joint.attrib["name"] for joint in model.findall("./JointSet/objects/*")])
        skeleton.GetJointNamesAttr().Set(jointNames)

        jointFramesDict = dict()
        jointOffsetsFramesDict = dict()
        jointParents = dict()
        joints=[]
        bindTransformsDict = dict()
        bindTransforms=[] # World space transform of each joint
        restTransforms=[] # Local space rest transforms of each joint, fallback for joints with no animation.
        for joint in model.findall("./JointSet/objects/*"):
            print("\tJoint Type:", joint.tag, "[", joint.attrib["name"], "]")
            parentFrame = joint.find("./socket_parent_frame").text
            childFrame = joint.find("./socket_child_frame").text
            jointFramesDict[joint.attrib["name"]] = (parentFrame, childFrame)
            for offsetFrame in joint.findall("./frames/PhysicalOffsetFrame"):
                offset = offsetFrame.attrib["name"]
                parent = offsetFrame.find("./socket_parent").text
                translation = Gf.Vec3d([float(x) for x in offsetFrame.find("./translation").text.split()])
                rotationsXYZ = [float(e) for e in offsetFrame.find("./orientation").text.split()]
                xRotation = Gf.Rotation(Gf.Vec3d([1.0, 0.0, 0.0]), math.degrees(rotationsXYZ[0]))
                yRotation = Gf.Rotation(Gf.Vec3d([0.0, 1.0, 0.0]), math.degrees(rotationsXYZ[1]))
                zRotation = Gf.Rotation(Gf.Vec3d([0.0, 0.0, 1.0]), math.degrees(rotationsXYZ[2]))
                orientation = xRotation * yRotation * zRotation
                print("\t\ttranslation: ", translation, "orientation: ", orientation)

                jointOffsetsFramesDict[offset] = (parent, translation, orientation)

            parentBody = os.path.basename(jointOffsetsFramesDict[jointFramesDict[joint.attrib["name"]][0]][0])
            childBody = os.path.basename(jointOffsetsFramesDict[jointFramesDict[joint.attrib["name"]][1]][0])
            jointParents[childBody] = parentBody

            parentSkelSpaceTransform = Gf.Matrix4d(1.0) # identity matrix
            if parentBody != "ground":
                parentSkelSpaceTransform = bindTransformsDict[parentBody]
            (parent, translation, orientation) = jointOffsetsFramesDict[parentFrame]
            parentJointSkelSpaceTransform = Gf.Matrix4d(orientation, translation)
            bindTransform = parentJointSkelSpaceTransform * parentSkelSpaceTransform
            bindTransforms.append(bindTransform)
            bindTransformsDict[childBody] = bindTransform

            # Add joint to joint hierarchy
            jointPath = childBody
            while(parentBody != "ground"):
                jointPath = parentBody + "/" + jointPath
                parentBody = jointParents[parentBody]
            joints.append(jointPath)

        jointPaths = Vt.TokenArray(joints)
        skeleton.GetJointsAttr().Set(jointPaths)

        bindTransformsArray = Vt.Matrix4dArray(bindTransforms)
        skeleton.GetBindTransformsAttr().Set(bindTransformsArray)

        for meshGeom in meshBodyDict:
            body = meshBodyDict[meshGeom]
            transformOp = meshGeom.AddTransformOp()
            transformOp.Set(bindTransformsDict[body])


        print("Joint Frame Dict: ", jointFramesDict)
        print("PhysicalOffsetFrame: ", jointOffsetsFramesDict)
        print("BindTransformsDict: ", bindTransformsDict)

        # Find mesh transforms from joint transform data.



        # TODO: Physical properties for simulation
        # Mass, center of mass and inertia tensor data per body

        # TODO: New muscle schema
        # TODO: New wrap object schema

        # Parse forces (like muscles)
        for force in model.findall("./ForceSet/objects/*"):
            print("\tForce Type:", force.tag, "[", force.attrib["name"],"]")

        # Parse marker set
        for markerset in model.findall("./MarkerSet"):
            print("\tMarker set:", markerset.attrib["name"])
            for marker in markerset.findall("./objects/Marker"):
                parentFrame = marker.find("./socket_parent_frame")
                location = marker.find("./location")
                localCoords = [float(x) for x in location.text.split()]
                print("\t\tMarker ", marker.attrib["name"], ": ", parentFrame.text, localCoords)

        stage.GetRootLayer().Save()
        stage.Export(usdPath + "a") # Save a usda file as well
        return usdPath

def osim2usd(osimPath, usdPath):

    print(f"Input OpenSim model path set to: {osimPath}")

    geomPath = os.path.dirname(osimPath) + "/Geometry"
    print(f"Input OpensSim geometry path set to: {geomPath}")
    print(f"Output USD scene path set to: {usdPath}")


    tree = xmlTree.parse(osimPath)
    usdPath = writeUsd(tree, usdPath, geomPath)

    return usdPath


def WriteAnimatedSkel(stage, skelPath, jointPaths,
                      rootTransformsPerFrame,
                      jointWorldSpaceTransformsPerFrame,
                      times, bindTransforms, restTransforms=None):
    if not len(rootTransformsPerFrame) == len(times):
        return False
    if not len(jointWorldSpaceTransformsPerFrame) == len(times):
        return False
    if not len(bindTransforms) == len(jointPaths):
        return False
    skel = UsdSkel.Skeleton.Define(stage, skelPath)
    if not skel:
        Tf.Warn("Failed defining a Skeleton at <%s>.", skelPath)
        return False
    numJoints = len(jointPaths)
    topo = UsdSkel.Topology(jointPaths)
    valid, whyNot = topo.Validate()
    if not valid:
        Tf.Warn("Invalid topology: %s" % reason)
        return False
    jointTokens = Vt.TokenArray([jointPath.pathString for jointPath in jointPaths])
    skel.GetJointsAttr().Set(jointTokens)
    skel.GetBindTransformsAttr().Set(bindTransforms)
    if restTransforms and len(restTransforms) == numJoints:
        skel.GetRestTransformsAttr().Set(restTransforms)
    rootTransformAttr = skel.MakeMatrixXform()
    for i, time in enumerate(times):
        rootTransformAttr.Set(rootTransformsPerFrame[i], time)
    anim = UsdSkel.Animation.Define(stage, skelPath.AppendChild("Anim"))
    binding = UsdSkel.BindingAPI.Apply(skel.GetPrim())
    binding.CreateSkeletonRel().SetTargets([anim.GetPrim().GetPath()])
    anim.GetJointsAttr().Set(jointTokens)
    for i, time in enumerate(times):
        rootTransform = rootTransformsPerFrame[i]
        jointWorldSpaceTransforms = jointWorldSpaceTransformsPerFrame[i]

        if len(jointWorldSpaceTransforms) == numJoints:

            jointLocalSpaceTransforms = \
                UsdSkel.ComputeJointLocalTransforms(
                    topo, jointWorldSpaceTransforms, rootTransform)
            if jointLocalSpaceTransforms:
                anim.SetTransforms(jointLocalSpaceTransforms, time)
    # Don't forget to call Save() on the stage!

    return True


def main(argv):

    print(f"OpenSim version: {osim.GetVersionAndDate()}")

    sessionPath=""
    modelPath = "./Model/LaiArnoldModified2017_poly_withArms_weldHand_scaled_adjusted.osim"
    inputPath = sessionPath + modelPath
    outputPath = os.path.splitext(inputPath)[0] + ".usd"

    opts, args = getopt.getopt(argv,"hi:o:",["input=","output="])
    for opt, arg in opts:
        if opt == "-h":
            print("osim2usd.py -i <inputFile> -o <outputFile>")
            sys.exit()
        elif opt in ("-i", "--input"):
            inputPath = arg
        elif opt in ("=o", "--output"):
            outputPath = arg

    usdPath = osim2usd(inputPath, outputPath)
    print(f"Saved usdPath to: {usdPath}")

# Checks if running this file from a script vs. a module. Useful if planning to use this file also as a module
# to incorporate into other scripts.
if __name__ == "__main__":
    main(sys.argv[1:])