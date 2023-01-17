# Library imports
import math, os, sys, getopt, numpy

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

class Function:
    def __init__(self, name):
        self.name = name

    def calcValue(self, x):
        return x

    def calcDerivative(self, x):
        return 1.0

class Constant(Function):
    def __init__(self, name, constantValue):
        self.constantValue = constantValue
        Function.__init__(self, name)

    def calcValue(self, x):
        return self.constantValue

    def calcDerivative(self, x):
        return 0.0
class SimmSpline(Function):
    def __init__(self, name, timeList, valueList):
        Function.__init__(self, name)
        assert(len(timeList) == len(valueList))
        self.simmSpline = osim.SimmSpline()
        for i in range(len(timeList)):
           self.simmSpline.addPoint(timeList[i], valueList[i])
    def calcValue(self, x):
        return self.simmSpline.calcValue(osim.Vector(1, x))
    def calcDerivative(self, x):
        return self.simmSpline.calcDerivative(x)

class LinearFunction(Function):
    def __init__(self, name, a, b):
        Function.__init__(self, name)
        self.a = a
        self.b = b

    def calcValue(self, x):
        return self.a*x + self.b

    def calcDerivative(self, x):
        return self.a

class PolynomialFunction(Function):

    def __init__(self, name, coeffs):
        Function.__init__(self, name)
        self.coeffs = coeffs

    def calcValue(self, x):
        maxDegree = len(self.coeffs) - 1
        degree = maxDegree
        value = 0.0
        for c in self.coeffs:
            value += c * pow(x, degree)
            degree = degree - 1
        return value

    def calcDerivative(self, x):
        maxDegree = len(self.coeffs) - 1
        degree = maxDegree
        value = 0.0
        for c in self.coeffs:
            if degree > 0:
                value += c * degree * pow(x, degree-1)
        return value
class MultiplierFunction(Function):

    def __init__(self, name, multiplier, function):
        Function.__init__(self, name)
        self.multiplier = multiplier
        self.function = function
    def calcValue(self, x):
        return self.multiplier * self.function.calcValue(x)

    def calcDerivative(self, x):
        return self.multiplier * self.function.calcDerivative(x)

class SpatialTransform:

    def __init__(self, joint):
        self.transformAxisDictionary = dict()
        print("Building spatial transform for joint: ", joint.attrib["name"])

        for transformAxis in joint.findall("./SpatialTransform/TransformAxis"):
            type = transformAxis.attrib["name"]
            axis = Gf.Vec3d([float(x) for x in transformAxis.find("./axis").text.split()])

            for functionType in ["./SimmSpline", "./Constant", "./LinearFunction", "./MultiplierFunction"]:
                function = transformAxis.find(functionType)
                if function == None:
                    continue

                axisFunction = self.createFunction(function)
                break

            self.addTransformAxis(type, axis, axisFunction)

    def createFunction(self, functionElement):
        if functionElement.tag == "SimmSpline":
            xList = [float(x) for x in functionElement.find("./x").text.split()]
            yList = [float(y) for y in functionElement.find("./y").text.split()]
            axisFunction = SimmSpline(functionElement.attrib["name"], xList, yList)
        elif functionElement.tag == "Constant":
            value = float(functionElement.find("./value").text)
            axisFunction = Constant(functionElement.attrib["name"], value)
        elif functionElement.tag == "LinearFunction":
            coeffs = [float(x) for x in functionElement.find("./coefficients").text.split()]
            axisFunction = LinearFunction(functionElement.attrib["name"], coeffs[0], coeffs[1])
        elif functionElement.tag == "MultiplierFunction":
            scale = float(functionElement.find("./scale").text)
            multiplierFunctionElement = functionElement.find("./function/*")
            multiplierFunction = self.createFunction(multiplierFunctionElement)
            axisFunction = MultiplierFunction(functionElement.attrib["name"], scale, multiplierFunction)
        elif functionElement.tag == "PolynomialFunction":
            coeffs = [float(x) for x in functionElement.find("./coefficients").text.split()]
            axisFunction = PolynomialFunction(functionElement.attrib["name"], coeffs)
        else:
            print("Error: Could not support function ", functionElement.tag)

        return axisFunction

    def addTransformAxis(self, type, axis, function):
        self.transformAxisDictionary[type] = (axis, function)

    def calcTransform(self, t):

        translation = Gf.Vec3d(0, 0, 0)
        orientation = Gf.Rotation()
        orientation.SetIdentity()
        for type in [ "translation1", "translation2", "translation3", "rotation1", "rotation2", "rotation3"]:
            if type in self.transformAxisDictionary:
                (axis, axisFunction) = self.transformAxisDictionary[type]
                # print("Type: ", type, "Axis: ", axis, "axisFunction:", axisFunction)
                if "translation" in type:
                    translation += (axisFunction.calcValue(t) * axis)
                if "rotation" in type: # Convert angles to degrees
                    orientation *= Gf.Rotation(axis, math.degrees(axisFunction.calcValue(t)))
        return (translation, orientation)

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



def writeUsd(parseTree, usdPath, geomPath, markerSpheres):
    root = parseTree.getroot()
    stage = Usd.Stage.CreateNew(usdPath)

    for model in root.findall("./Model"):
        print("Model: ", model.attrib["name"])
        skelRootPath = "/" + model.attrib["name"]
        modelPrim = UsdSkel.Root.Define(stage, skelRootPath)
        stage.SetDefaultPrim(stage.GetPrimAtPath(skelRootPath))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

        meshBodyDict = dict()
        for bodyset in model.findall("./BodySet"):
            print("Bodyset: ", bodyset.attrib["name"])
            skeletonPath = skelRootPath + "/" + bodyset.attrib["name"]
            skeleton = UsdSkel.Skeleton.Define(stage, skeletonPath)

            bodyIndex = 0
            bodyName2Index = dict()
            for body in bodyset.findall("./objects/Body"):
                print("\tBody: ", body.attrib["name"])
                bodyName2Index[body.attrib["name"]] = bodyIndex

                for mesh in body.findall("./attached_geometry/Mesh"):
                    print("\t\tMesh: ", mesh.attrib["name"])

                    meshPath = skelRootPath + "/meshes/" + mesh.attrib["name"]
                    meshGeom = UsdGeom.Mesh.Define(stage, meshPath)
                    meshPrim = stage.GetPrimAtPath(meshPath)

                    # Sets up binding of this mesh to a joint.
                    binding = UsdSkel.BindingAPI.Apply(meshPrim)
                    jointIndicesPrimvar = binding.CreateJointIndicesPrimvar(True)
                    jointWeightsPrimvar = binding.CreateJointWeightsPrimvar(True)
                    binding.SetRigidJointInfluence(bodyIndex, 1.0)

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
                    # print("\t\t\tMeshFile: ", meshFile.text)

                    meshPath = geomPath + "/" + meshFile.text
                    ( faceVertexCounts, faceVertexIndices, points ) = getMeshArrays(meshPath)

                    faceVertexCountsAttr = meshPrim.CreateAttribute('faceVertexCounts', Sdf.ValueTypeNames.IntArray)
                    faceVertexCountsAttr.Set(Vt.IntArray(faceVertexCounts))

                    faceVertexIndicesAttr = meshPrim.CreateAttribute('faceVertexIndices', Sdf.ValueTypeNames.IntArray)
                    faceVertexIndicesAttr.Set(Vt.IntArray(faceVertexIndices))

                    pointsAttr = meshPrim.CreateAttribute('points', Sdf.ValueTypeNames.Float3Array)
                    pointsAttr.Set(Vt.Vec3fArray.FromNumpy(points))
                bodyIndex = bodyIndex + 1
                # End Body Loop

            for wrappedObject in body.findall("./WrapObjectSet/objects/*"):
                print("\t\t WrappedObject[", wrappedObject.tag, "] = ", wrappedObject.attrib["name"])

        # print("meshBodyDict: ", meshBodyDict)
        # print("bodyName2Index: ", bodyName2Index)

        # Parse joints
        jointNames = Vt.TokenArray([joint.attrib["name"] for joint in model.findall("./JointSet/objects/*")])
        skeleton.GetJointNamesAttr().Set(jointNames)

        jointFramesDict = dict()
        bodyJointOffsetDict = dict()
        jointParents = dict()
        joints=[]
        bindTransformsDict = dict()
        bindTransforms=[] # World space transform of each joint
        restTransforms=[] # Local space rest transforms of each joint, fallback for joints with no animation.
        for joint in model.findall("./JointSet/objects/*"):
            # print("\tJoint Type:", joint.tag, "[", joint.attrib["name"], "]")
            parentFrame = joint.find("./socket_parent_frame").text
            childFrame = joint.find("./socket_child_frame").text
            jointFramesDict[joint.attrib["name"]] = (parentFrame, childFrame)
            jointOffsetsFramesDict = dict()
            for offsetFrame in joint.findall("./frames/PhysicalOffsetFrame"):
                offset = offsetFrame.attrib["name"]
                parent = os.path.basename(offsetFrame.find("./socket_parent").text)
                translation = Gf.Vec3d([float(x) for x in offsetFrame.find("./translation").text.split()])
                rotationsXYZ = [float(e) for e in offsetFrame.find("./orientation").text.split()]
                xRotation = Gf.Rotation(Gf.Vec3d([1.0, 0.0, 0.0]), math.degrees(rotationsXYZ[0]))
                yRotation = Gf.Rotation(Gf.Vec3d([0.0, 1.0, 0.0]), math.degrees(rotationsXYZ[1]))
                zRotation = Gf.Rotation(Gf.Vec3d([0.0, 0.0, 1.0]), math.degrees(rotationsXYZ[2]))
                orientation = xRotation * yRotation * zRotation
                jointOffsetsFramesDict[offset] = (parent, translation, orientation)

            spatialTranslation = Gf.Vec3d(0, 0, 0)
            spatialOrientation = Gf.Rotation().SetIdentity()
            if joint.tag == "CustomJoint":
                spatialTransform = SpatialTransform(joint)
                (spatialTranslation, spatialOrientation) = spatialTransform.calcTransform(0.0)
                # print("SpatialTransform = ", spatialTranslation, spatialOrientation)
            # Compute custom joint offset, bake it into the bind transform
            localSpatialTransform = Gf.Matrix4d(spatialOrientation, spatialTranslation)

            parentBody = os.path.basename(jointOffsetsFramesDict[jointFramesDict[joint.attrib["name"]][0]][0])
            childBody = os.path.basename(jointOffsetsFramesDict[jointFramesDict[joint.attrib["name"]][1]][0])

            (childParent, childTranslation, childOrientation) = jointOffsetsFramesDict[childFrame]
            assert(childParent == childBody)
            bodyJointOffsetDict[childBody] = (childTranslation, childOrientation)
            jointParents[childBody] = parentBody

            parentSkelSpaceTransform = Gf.Matrix4d(1.0) # identity matrix
            invParentOffsetTransform = Gf.Matrix4d(1.0)
            if parentBody != "ground":
                (parentOffsetTranslation, parentOffsetOrientation) = bodyJointOffsetDict[parentBody]
                invParentOffsetTransform = Gf.Matrix4d(parentOffsetOrientation, parentOffsetTranslation).GetInverse()
                parentSkelSpaceTransform = bindTransformsDict[parentBody]


            (parent, translation, orientation) = jointOffsetsFramesDict[parentFrame]
            # Find inboard offset joint and use to adjust bone geometry reference frame.
            parentJointSkelSpaceTransform = Gf.Matrix4d(orientation, translation)
            bindTransform = localSpatialTransform * parentJointSkelSpaceTransform * invParentOffsetTransform * parentSkelSpaceTransform
            bindTransforms.append(bindTransform)

            restTransform = localSpatialTransform * parentJointSkelSpaceTransform * invParentOffsetTransform
            restTransforms.append(restTransform)

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

        restTransformsArray = Vt.Matrix4dArray(restTransforms)
        skeleton.GetRestTransformsAttr().Set(restTransformsArray)

        for meshGeom in meshBodyDict:
            body = meshBodyDict[meshGeom]
            (inboardTranslation, inboardOrientation) = bodyJointOffsetDict[body]
            invInboardTransform = Gf.Matrix4d(inboardOrientation, inboardTranslation).GetInverse()
            transformOp = meshGeom.AddTransformOp()
            transformOp.Set(invInboardTransform * bindTransformsDict[body])

            # We previously added a scale operation, but this must be applied the most locally, so the order needs to be changed in the transform list.
            # Reverse order of transforms, so that scale is the most local transform (last in list)
            transformList = meshGeom.GetOrderedXformOps()
            transformList.reverse()
            meshGeom.SetXformOpOrder(transformList)

        #print("Joint parents: ", jointParents)
        #print("Joint Frame Dict: ", jointFramesDict)
        #print("PhysicalOffsetFrame: ", jointOffsetsFramesDict)
        #print("BindTransformsDict: ", bindTransformsDict)

        # Find mesh transforms from joint transform data.



        # TODO: Physical properties for simulation
        # Mass, center of mass and inertia tensor data per body

        # TODO: New muscle schema
        # TODO: New wrap object schema

        # Parse forces (like muscles)
        #for force in model.findall("./ForceSet/objects/*"):
        #    print("\tForce Type:", force.tag, "[", force.attrib["name"],"]")

        # Parse marker set
        for markerset in model.findall("./MarkerSet"):
            #print("\tMarker set:", markerset.attrib["name"])
            for marker in markerset.findall("./objects/Marker"):
                parentFrame = marker.find("./socket_parent_frame")
                location = marker.find("./location")
                localCoords = [float(x) for x in location.text.split()]
                #print("\t\tMarker ", marker.attrib["name"], ": ", parentFrame.text, localCoords)
                # Replace . in names with _ for proper USD primitive path names.
                if markerSpheres == True:
                    markerGeom = UsdGeom.Sphere.Define(stage, skelRootPath + "/" + markerset.attrib["name"] + "/" + marker.attrib["name"].replace(".", "_"))
                    markerGeom.GetRadiusAttr().Set(0.01)
                else:
                    markerGeom = UsdGeom.Cube.Define(stage, skelRootPath + "/" + markerset.attrib["name"] + "/" + marker.attrib["name"].replace(".","_"))
                    markerGeom.GetSizeAttr().Set(0.03)


                markerGeom.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])

                markerLocationOp = markerGeom.AddTranslateOp()
                markerLocationOp.Set(Gf.Vec3f(localCoords))
                markerTransformOp = markerGeom.AddTransformOp()
                body = os.path.basename(parentFrame.text)
                markerTransform = bindTransformsDict[body]
                markerTransformOp.Set(markerTransform)

                # Sets up binding of this mesh to a joint.
                binding = UsdSkel.BindingAPI.Apply(markerGeom.GetPrim())
                jointIndicesPrimvar = binding.CreateJointIndicesPrimvar(True)
                jointWeightsPrimvar = binding.CreateJointWeightsPrimvar(True)
                bodyIndex = bodyName2Index[body]
                binding.SetRigidJointInfluence(bodyIndex, 1.0)

        stage.GetRootLayer().Save()
        stage.Export(usdPath + "a") # Save a usda file as well
        return usdPath

def osim2usd(osimPath, usdPath, markerSpheres):

    print(f"Input OpenSim model path set to: {osimPath}")

    geomPath = os.path.dirname(osimPath) + "/Geometry"
    print(f"Input OpensSim geometry path set to: {geomPath}")
    print(f"Output USD scene path set to: {usdPath}")


    tree = xmlTree.parse(osimPath)
    usdPath = writeUsd(tree, usdPath, geomPath, markerSpheres)

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
    markersAsSpheres = True

    opts, args = getopt.getopt(argv,"hi:o:",["input=","output="])
    for opt, arg in opts:
        if opt == "-h":
            print("osim2usd.py -i <inputFile> -o <outputFile> [-m <markerStyle>]")
            sys.exit()
        elif opt in ("-i", "--input"):
            inputPath = arg
        elif opt in ("-o", "--output"):
            outputPath = arg
        elif opt in ("-m", "--markers"):
            if arg == "spheres":
                markersAsSpheres = True

    usdPath = osim2usd(inputPath, outputPath, markersAsSpheres)
    print(f"Saved usdPath to: {usdPath}")

# Checks if running this file from a script vs. a module. Useful if planning to use this file also as a module
# to incorporate into other scripts.
if __name__ == "__main__":
    main(sys.argv[1:])