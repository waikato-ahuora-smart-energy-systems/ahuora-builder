from idaes.core.base.property_base import PhysicalParameterBlock
from idaes.core import FlowsheetBlock
from property_packages.build_package import build_package
from property_packages.modular.modular_extended import GenericExtendedParameterBlock
from ahuora_builder_types.flowsheet_schema import PropertyPackageType
from .custom_property_package_converter import encapsulate_custom_property_package
from .flowsheet_manager_type import FlowsheetManager
from ahuora_builder_types import PropertyPackageId


def create_property_package(
    property_package_schema: PropertyPackageType, flowsheet: FlowsheetBlock
) -> PhysicalParameterBlock:
    """
    Create a property package from a schema
    """
    if property_package_schema.custom:
        idaes_configuration = encapsulate_custom_property_package(property_package_schema.custom_properties,
                                                                  property_package_schema.compounds_properties)
        property_package = GenericExtendedParameterBlock(**idaes_configuration)
    else:
        compounds = property_package_schema.compounds
        type = property_package_schema.type
        phases = property_package_schema.phases

        property_package = build_package(type, compounds)
    property_package_id = property_package_schema.id if property_package_schema.id != -1 else "default"

    flowsheet.add_component(f"PP_{property_package_id}", property_package)
    return property_package


class PropertyPackageManager:
    """
    Manages the property packages for a flowsheet
    """

    def __init__(self, flowsheet_manager: FlowsheetManager) -> None:
        """
        Create a new property package manager
        """
        self._flowsheet_manager = flowsheet_manager
        self._property_packages: dict[PropertyPackageId, PhysicalParameterBlock] = {}

    def load(self) -> None:
        """
        Load the property packages from the flowsheet definition
        """
        schema = self._flowsheet_manager.schema
        property_packages_schema = schema.property_packages
        for property_package_schema in property_packages_schema:
            id = property_package_schema.id
            if id in self._property_packages:
                raise Exception(f"Property package with id {id} already exists")

            property_package = create_property_package(
                property_package_schema, self._flowsheet_manager.model.fs
            )
            self._property_packages[id] = property_package

    def get(self, id: PropertyPackageId) -> PhysicalParameterBlock:
        """
        Get a property package by id
        """
        # For backwards compatibility with other tests, use id -1 as helmholtz
        fs = self._flowsheet_manager.model.fs

        if id == -1:
            if not hasattr(fs, "PP_default"):
                create_property_package(
                    PropertyPackageType(
                        id=-1,
                        type="helmholtz",
                        compounds=["h2o"],
                        phases=["Liq"]
                    ),
                    fs,
                )
            return fs.PP_default
        else:
            # get the property package by id
            if id not in self._property_packages:
                raise Exception(f"Property package with id {id} does not exist")
            return self._property_packages[id]
