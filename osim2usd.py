# Library imports
import math, os, sys, getopt, numpy

# For XML parsing
import xml.etree.ElementTree as xmlTree

# For OpenSim support
import opensim as osim

# For USD support
from pxr import Sdf, Gf, Usd, UsdGeom, UsdSkel, Vt

# For vtk support (reading vtp geometry)
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# For mot file support
from mot2quats import mot2quats

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

def writeUSDAnimations(skeleton, modelPath, animations, optionsDict):
    print("Writing USD Animations...")
    # Build a snapshot to query
    skeletonCache = UsdSkel.Cache()
    skeletonQuery = skeletonCache.GetSkelQuery(skeleton)

    for animation in animations:
        (motionName, motionPoses) = animation
        print("\tAnimation: ", motionName)

        animationName = optionsDict["outputFolder"] + "/" + motionName + "." + optionsDict["format"]
        animStage = Usd.Stage.CreateNew(animationName)
        UsdGeom.SetStageUpAxis(animStage, UsdGeom.Tokens.y)

        motionPrim = animStage.DefinePrim("/motion")

        motionPrim.GetReferences().AddReference(modelPath)

        animStage.SetDefaultPrim(motionPrim)
        binding = UsdSkel.BindingAPI.Apply(motionPrim)
        animNode = UsdSkel.Animation.Define(animStage, Sdf.Path("/motion/"+motionName))

        # May not be needed if have multiple anims
        binding.CreateAnimationSourceRel().SetTargets([animNode.GetPrim().GetPath()])

        # GetTopology() returns a UsdSkel.Topology object, which describes
        # the parent<->child relationships. It also gives the number of joints.
        topology = skeletonQuery.GetTopology()
        numJoints = len(topology)
        jointOrder = skeletonQuery.GetJointOrder()

        # Build up the joint attributes list based on contents of bodyNames
        (bodyNames, times, poseTrajectories) = motionPoses

        animStage.SetStartTimeCode(1)
        animStage.SetEndTimeCode(len(times))

        jointPaths = []
        for i in range(numJoints):
            jointPath = Sdf.Path(jointOrder[i])
            name = jointPath.name
            if name in bodyNames:
                jointPaths.append(jointPath)
            parent = topology.GetParent(i)

        jointTokens = Vt.TokenArray([jointPath.pathString for jointPath in jointPaths])
        animNode.GetJointsAttr().Set(jointTokens)
        rotationsAttr = animNode.CreateRotationsAttr()

        for i, time in enumerate(times):
            bodyposes = poseTrajectories[i]
            quats = []
            for bodypose in bodyposes:
                rotation = bodypose[1]
                quats.append(Gf.Quatf(rotation.w, rotation.x, rotation.y, rotation.z))
            jointQuats = Vt.QuatfArray(quats)
            rotationsAttr.Set(time=i, value=jointQuats)

        # Save out animation file.
        animStage.GetRootLayer().Save()
        print("Saved to: ", animationName)


def writeUsd(parseTree, geomPath, osimPath, motionFiles, optionsDict):
    root = parseTree.getroot()
    usdPath = os.path.splitext(os.path.basename(osimPath))[0] + "." + optionsDict["format"]
    stage = Usd.Stage.CreateNew(usdPath)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetMetadata('comment', 'Generated with osim2usd.py by digitalmonkey')

    for model in root.findall("./Model"):
        print("Model: ", model.attrib["name"])
        skelRootPath = "/" + model.attrib["name"]
        skelRoot = UsdSkel.Root.Define(stage, skelRootPath)

        stage.SetDefaultPrim(stage.GetPrimAtPath(skelRootPath))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

        # Add Skeleton
        skeletonPath = skelRootPath + "/" + model.attrib["name"]
        skeleton = UsdSkel.Skeleton.Define(stage, skeletonPath)
        binding = UsdSkel.BindingAPI.Apply(skeleton.GetPrim())

        meshBodyDict = dict()
        wrapBodyDict = dict()
        for bodyset in model.findall("./BodySet"):
            bodysetPath = skelRootPath + "/" + bodyset.attrib["name"]
            bodysetXform = UsdGeom.Xform.Define(stage, bodysetPath)

            bodyIndex = 0
            bodyName2Index = dict()
            for body in bodyset.findall("./objects/Body"):
                bodyName2Index[body.attrib["name"]] = bodyIndex

                # Process mesh geometry.
                for mesh in body.findall("./attached_geometry/Mesh"):
                    meshPath = skelRootPath + "/" + bodyset.attrib["name"] + "/" + mesh.attrib["name"]
                    meshGeom = UsdGeom.Mesh.Define(stage, meshPath)
                    meshPrim = stage.GetPrimAtPath(meshPath)

                    scaleFactor = mesh.find("./scale_factors")
                    scaleFactors = [float(x) for x in scaleFactor.text.split()]
                    meshBodyDict[meshGeom] = (body.attrib["name"], scaleFactors)

                    color = mesh.find("./Appearance/color")
                    colors = [float(x) for x in color.text.split()]
                    colorAttr = meshGeom.GetDisplayColorAttr()
                    colorAttr.Set([tuple(colors)])

                    opacity = mesh.find("./Appearance/opacity")
                    opacityValue = float(opacity.text)
                    opacityAttr = meshGeom.GetDisplayOpacityAttr()
                    opacityAttr.Set([opacityValue])

                    meshFile = mesh.find("./mesh_file")

                    meshFilePath = geomPath + "/" + meshFile.text
                    ( faceVertexCounts, faceVertexIndices, points ) = getMeshArrays(meshFilePath)

                    faceVertexCountsAttr = meshPrim.CreateAttribute('faceVertexCounts', Sdf.ValueTypeNames.IntArray)
                    faceVertexCountsAttr.Set(Vt.IntArray(faceVertexCounts))

                    faceVertexIndicesAttr = meshPrim.CreateAttribute('faceVertexIndices', Sdf.ValueTypeNames.IntArray)
                    faceVertexIndicesAttr.Set(Vt.IntArray(faceVertexIndices))

                    pointsAttr = meshPrim.CreateAttribute('points', Sdf.ValueTypeNames.Float3Array)
                    pointsAttr.Set(Vt.Vec3fArray.FromNumpy(points))

                # Process wrap objects.
                if optionsDict["wrapObjects"] == True:
                    for wrappedObject in body.findall("./WrapObjectSet/objects/*"):
                        print("\t\tWrappedObject[", wrappedObject.tag, "] =", wrappedObject.attrib["name"])
                        if wrappedObject.tag == "WrapCylinder":
                            wrapPath = skelRootPath + "/" + bodyset.attrib["name"] + "/" + wrappedObject.attrib["name"]
                            wrapCylinder = UsdGeom.Cylinder.Define(stage, wrapPath)

                            # Set bind transform
                            wrapRotationXYZ = Gf.Vec3d([float(c) for c in wrappedObject.find("./xyz_body_rotation").text.split()])
                            wrapTranslation = Gf.Vec3d([float(c) for c in wrappedObject.find("./translation").text.split()])
                            xRotation = Gf.Rotation(Gf.Vec3d([1.0, 0.0, 0.0]), math.degrees(wrapRotationXYZ[0]))
                            yRotation = Gf.Rotation(Gf.Vec3d([0.0, 1.0, 0.0]), math.degrees(wrapRotationXYZ[1]))
                            zRotation = Gf.Rotation(Gf.Vec3d([0.0, 0.0, 1.0]), math.degrees(wrapRotationXYZ[2]))
                            wrapOrientation = xRotation * yRotation * zRotation
                            wrapTransform = Gf.Matrix4d(wrapOrientation, wrapTranslation)
                            wrapActive = wrappedObject.find("./active").text
                            if wrapActive == "false":
                                wrapCylinder.SetActive(False)
                            wrapQuadrant = wrappedObject.find("./quadrant").text
                            wrapCylinder.GetPrim().CreateAttribute("quadrant", Sdf.ValueTypeNames.String).Set(wrapQuadrant)
                            wrapColor = Gf.Vec3d([float(c) for c in wrappedObject.find("./Appearance/color").text.split()])
                            wrapCylinder.GetDisplayColorAttr().Set([(wrapColor[0], wrapColor[1], wrapColor[2])])
                            wrapOpacity = float(wrappedObject.find("./Appearance/opacity").text)
                            wrapCylinder.GetDisplayOpacityAttr().Set([wrapOpacity])
                            wrapCylinder.GetAxisAttr().Set("Z")
                            wrapRadius = float(wrappedObject.find("./radius").text)
                            wrapCylinder.GetRadiusAttr().Set(wrapRadius)
                            wrapLength = float(wrappedObject.find("./length").text)
                            wrapCylinder.GetHeightAttr().Set(wrapLength)
                            wrapBodyDict[wrapCylinder] = (body.attrib["name"], wrapTransform)

                bodyIndex = bodyIndex + 1
                # End Body Loop

        # Parse joints: Not needed
        if optionsDict["jointNames"] == True:
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

            (inboardTranslation, inboardOrientation) = bodyJointOffsetDict[childBody]
            inboardTransform = Gf.Matrix4d(inboardOrientation, inboardTranslation)
            invInboardTransform = inboardTransform.GetInverse()
            parentInboardTransform = Gf.Matrix4d().SetIdentity()
            if parentBody != "ground":
                (parentInboardTranslation, parentInboardOrientation) = bodyJointOffsetDict[parentBody]
                parentInboardTransform = Gf.Matrix4d(parentInboardOrientation, parentInboardTranslation)

            bindTransform = invInboardTransform * localSpatialTransform * parentJointSkelSpaceTransform * invParentOffsetTransform * parentInboardTransform * parentSkelSpaceTransform
            bindTransforms.append(bindTransform)

            restTransform = invInboardTransform * localSpatialTransform * parentJointSkelSpaceTransform * invParentOffsetTransform * parentInboardTransform
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
            (body, scalefactors) = meshBodyDict[meshGeom]

            # Sets up binding of this mesh to a joint.
            binding = UsdSkel.BindingAPI.Apply(meshGeom.GetPrim())
            binding.CreateSkeletonRel().SetTargets([skeleton.GetPrim().GetPath()])
            bodyIndex = bodyName2Index[body]
            binding.SetRigidJointInfluence(bodyIndex, 1.0)

            # Set up geometry transform for binding
            geomBindAttr = binding.CreateGeomBindTransformAttr()
            scaleTransform = Gf.Matrix4d().GetInverse().SetScale(scalefactors)
            geomBindAttr.Set(scaleTransform * bindTransformsDict[body])

        # Adjust geometry binding transform to take into account inboard translations
        if optionsDict["wrapObjects"] == True:
            for wrapGeom in wrapBodyDict:
                (body, wrapTransform) = wrapBodyDict[wrapGeom]

                binding = UsdSkel.BindingAPI.Apply(wrapGeom.GetPrim())
                binding.CreateSkeletonRel().SetTargets([skeleton.GetPrim().GetPath()])
                jointIndicesPrimvar = binding.CreateJointIndicesPrimvar(True)
                jointWeightsPrimvar = binding.CreateJointWeightsPrimvar(True)

                bodyIndex = bodyName2Index[body]
                binding.SetRigidJointInfluence(bodyIndex, 1.0)

                (inboardTranslation, inboardOrientation) = bodyJointOffsetDict[body]
                invInboardTransform = Gf.Matrix4d(inboardOrientation, inboardTranslation).GetInverse()
                geomBindAttr = binding.CreateGeomBindTransformAttr()
                geomBindAttr.Set(wrapTransform * invInboardTransform * bindTransformsDict[body])


        # TODO: Physical properties for simulation
        # Mass, center of mass and inertia tensor data per body

        # Parse forces (like muscles)
        if optionsDict["muscles"] == True:
            muscles = model.findall("./ForceSet/objects/Millard2012EquilibriumMuscle")
            muscleSet = UsdGeom.BasisCurves.Define(stage, skelRootPath + "/muscles")
            muscleSet.GetTypeAttr().Set("linear")
            muscleSet.GetWrapAttr().Set("nonperiodic")
            muscleSet.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])

            pointsAttr = muscleSet.CreatePointsAttr()
            widthsAttr = muscleSet.CreateWidthsAttr()

            muscleCounts = []
            musclePoints = []
            widths = []
            for muscle in muscles:
                points = muscle.findall("./GeometryPath/PathPointSet/objects/*")
                muscleCounts.append(len(points))
                print("\t\tMuscle:", muscle.tag, "[", muscle.attrib["name"], "] = ", len(points), " points.")
                for point in points:
                    parentFrame = point.find("./socket_parent_frame").text
                    body = os.path.basename(parentFrame)
                    bodyTransform = bindTransformsDict[body]
                    location = [float(x) for x in point.find("./location").text.split()]
                    musclePoint = bodyTransform.Transform(Gf.Vec3f(location))
                    musclePoints.append(musclePoint)
                    widths.append(0.1)
            muscleSet.GetCurveVertexCountsAttr().Set(muscleCounts)
            pointsAttr.Set(Vt.Vec3fArray(musclePoints))
            widthsAttr.Set(widths)

        # Parse marker set
        if optionsDict["exportMarkers"] == True:
            for markerset in model.findall("./MarkerSet"):
                #print("\tMarker set:", markerset.attrib["name"])
                markersetPath = skelRootPath + "/" + markerset.attrib["name"]
                markersetXform = UsdGeom.Xform.Define(stage, markersetPath)

                for marker in markerset.findall("./objects/Marker"):
                    parentFrame = marker.find("./socket_parent_frame")
                    location = marker.find("./location")
                    localCoords = [float(x) for x in location.text.split()]
                    #print("\t\tMarker ", marker.attrib["name"], ": ", parentFrame.text, localCoords)
                    # Replace . in names with _ for proper USD primitive path names.
                    markerScale = optionsDict["markerSize"]
                    if optionsDict["markerSpheres"] == True:
                        markerGeom = UsdGeom.Sphere.Define(stage, skelRootPath + "/" + markerset.attrib["name"] + "/" + marker.attrib["name"].replace(".", "_"))
                    else:
                        markerGeom = UsdGeom.Cube.Define(stage, skelRootPath + "/" + markerset.attrib["name"] + "/" + marker.attrib["name"].replace(".","_"))

                    markerGeom.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
                    body = os.path.basename(parentFrame.text)
                    markerTransform = bindTransformsDict[body]

                    # Sets up binding of this marker to a joint.
                    binding = UsdSkel.BindingAPI.Apply(markerGeom.GetPrim())

                    # Bind marker to skeleton
                    binding.CreateSkeletonRel().SetTargets([skeleton.GetPrim().GetPath()])
                    jointIndicesPrimvar = binding.CreateJointIndicesPrimvar(True)
                    jointWeightsPrimvar = binding.CreateJointWeightsPrimvar(True)

                    bodyIndex = bodyName2Index[body]
                    binding.SetRigidJointInfluence(bodyIndex, 1.0)

                    # Set geometry bind transform in world space
                    geomBindAttr = binding.CreateGeomBindTransformAttr()
                    markerScaleTransform = Gf.Matrix4d().SetIdentity().SetScale([markerScale, markerScale, markerScale])
                    localMarkerTransform = Gf.Matrix4d().SetIdentity().SetTranslate(localCoords)
                    geomBindAttr.Set(markerScaleTransform * markerTransform * localMarkerTransform)

        # Saving layers in specified format.
        stage.GetRootLayer().Save()

        # Save out animations into separate usd files.
        animations = []
        for motion in motionFiles:
            (motionName, motionPoses) = mot2quats(motion, usdPath, jointParents, osimPath, optionsDict)
            animations.append((motionName, motionPoses))

        if len(animations) > 0:
            writeUSDAnimations(skeleton, usdPath, animations, optionsDict)
        return usdPath

def osim2usd(osimPath, motionFiles, optionsDict):

    print(f"Input OpenSim model path set to: {osimPath}")

    geomPath = os.path.dirname(osimPath) + "/Geometry"
    print(f"Input OpensSim geometry path set to: {geomPath}")

    tree = xmlTree.parse(osimPath)
    usdPath = writeUsd(tree, geomPath, osimPath, motionFiles, optionsDict)

    # We rename file here to move to output folder, while making sure the root layer
    # does not reference the output folder than if we passed the full path when we create the stage.
    newPath = optionsDict["outputFolder"] + "/" + usdPath
    os.rename(usdPath, newPath)
    return newPath

def main(argv):

    print(f"OpenSim version: {osim.GetVersionAndDate()}")

    sessionPath=""
    modelPath = "./Model/LaiArnoldModified2017_poly_withArms_weldHand_scaled_adjusted.osim"
    inputPath = sessionPath + modelPath
    optionsDict = dict()
    optionsDict["markerSpheres"] = False
    optionsDict["exportMarkers"]  = False
    optionsDict["markerSize"] = 0.01
    optionsDict["jointNames"] = True
    optionsDict["wrapObjects"] = False
    optionsDict["muscles"] = False
    optionsDict["format"] = "usda"
    optionsDict["outputFolder"] = "./Outputs"
    optionsDict["motionFormat"] = "localRotationsOnly"

    motionFiles = []

    opts, args = getopt.getopt(argv,"hi:o:",["input=","output="])
    for opt, arg in opts:
        if opt == "-h":
            print("osim2usd.py -i <inputFile> -o <outputFolder> -a <motionFile> [-m <markerStyle> -s <markerSize> -t <usd|usdc|usda>]")
            sys.exit()
        elif opt in ("-i", "--input", "--model"):
            inputPath = arg
        elif opt in ("-j", "--jointNames"):
            if arg == "1":
                optionsDict["jointNames"] = True
            else:
                optionsDict["jointNames"] = False
        elif opt in ("-o", "--outputFolder"):
            optionsDict["outputFolder"] = arg
        elif opt in ("-m", "--markers"):
            if arg == "spheres":
                optionsDict["markerSpheres"] = True
            elif arg == "none":
                optionsDict["exportMarkers"] = False
        elif opt in ("-s", "--markerSize"):
            options["markerSize"] = float(arg)
        elif opt in ("-a", "--motion", "--animation"):
            motionFiles = [ arg ]
        elif opt in ("--motionFolder", "-f"):
            motionFiles = os.listdir( arg )
        elif opt in ("-t", "--type"):
            optionsDict["format"] = arg

    motionFiles = [
        "./Motions/kinematics_activations_left_leg_squat_0.mot"
    ]

    optionsDict["columnsInDegrees"] = [
        "pelvis_tilt",
        "pelvis_list",
        "pelvis_rotation",
        "hip_flexion_l",
        "hip_adduction_l",
        "hip_rotation_l",
        "hip_flexion_r",
        "hip_adduction_r",
        "hip_rotation_r",
        "knee_angle_l",
        "knee_angle_r",
        "ankle_angle_l",
        "ankle_angle_r",
        "subtalar_angle_l",
        "subtalar_angle_r",
        "mtp_angle_l",
        "mtp_angle_r",
        "lumbar_extension",
        "lumbar_bending",
        "lumbar_rotation",
        "arm_flex_l",
        "arm_add_l",
        "arm_rot_l",
        "arm_flex_r",
        "arm_add_r",
        "arm_rot_r",
        "elbow_flex_l",
        "elbow_flex_r",
        "pro_sup_l",
        "pro_sup_r"
    ]

    if len(motionFiles) > 0:
        print("Motions to process: ", motionFiles)

    usdPath = osim2usd(inputPath, motionFiles, optionsDict)
    print(f"Saved usdPath to: {usdPath}")

# Checks if running this file from a script vs. a module. Useful if planning to use this file also as a module
# to incorporate into other scripts.
if __name__ == "__main__":
    main(sys.argv[1:])