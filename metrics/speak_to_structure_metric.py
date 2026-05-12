"""Molecular-structure validity and similarity metrics for Speak-to-Structure."""

import json
import re
from typing import Any, Dict, List, Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


_ATOM_FIELDS = [
    "carbon",
    "oxygen",
    "nitrogen",
    "sulfur",
    "fluorine",
    "chlorine",
    "bromine",
    "iodine",
    "phosphorus",
    "boron",
    "silicon",
    "selenium",
    "tellurium",
    "arsenic",
    "antimony",
    "bismuth",
    "polonium",
]

_FUNCTIONAL_GROUP_FIELDS = [
    "benzene rings",
    "hydroxyl",
    "anhydride",
    "aldehyde",
    "ketone",
    "carboxyl",
    "ester",
    "amide",
    "amine",
    "nitro",
    "halo",
    "thioether",
    "nitrile",
    "thiol",
    "sulfide",
    "disulfide",
    "sulfoxide",
    "sulfone",
    "borane",
]


def _correct_text(text: str) -> str:
    """Extract a likely SMILES string from model output using the upstream S2-Bench cleanup rules."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            json_str = match.group().replace('""', '"')
            try:
                obj = json.loads(json_str)
                candidate = obj.get("molecule") or obj.get("smiles") or json_str
            except Exception:
                candidate = json_str.split(":", 1)[1].strip().strip("}").strip().strip('"').strip()
            for sep in ["=>", "->"]:
                if sep in candidate:
                    candidate = candidate.split(sep)[-1].strip()
            return candidate.strip()

        # Prefer an explicit final-answer line if the model still emits prose.
        marker_patterns = [
            r"(?:^|\n)\s*Molecule\s*:\s*([^\n`]+)",
            r"(?:^|\n)\s*SMILES\s*:\s*([^\n`]+)",
            r"(?:^|\n)\s*Final answer\s*:\s*([^\n`]+)",
        ]
        for pattern in marker_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                candidate = matches[-1].strip().strip('"').strip("'").strip()
                for sep in ["=>", "->"]:
                    if sep in candidate:
                        candidate = candidate.split(sep)[-1].strip()
                return candidate

        candidate = text.replace("\n", " ").strip()
        for sep in ["=>", "->"]:
            if sep in candidate:
                candidate = candidate.split(sep)[-1].strip()
        if len(candidate) >= 2 and candidate[0] == "[" and candidate[-1] == "]":
            candidate = candidate[1:-1]
        if candidate.lower().startswith("molecule:"):
            candidate = candidate.split(":", 1)[1].strip()
        elif candidate.lower().startswith("smiles:"):
            candidate = candidate.split(":", 1)[1].strip()
        return candidate
    except Exception:
        return "None"


def _mol_from_output(text: str) -> Optional[Chem.Mol]:
    smiles = _correct_text(text)
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _mol_prop(mol: Chem.Mol, prop: str) -> Optional[float]:
    if mol is None:
        return None

    if prop == "validity":
        return 1.0
    if prop == "logP":
        return Descriptors.MolLogP(mol)
    if prop == "MR":
        return Descriptors.MolMR(mol)
    if prop == "qed":
        return Descriptors.qed(mol)
    if prop == "rot_bonds":
        return Descriptors.NumRotatableBonds(mol)
    if prop == "num_single_bonds":
        return sum(bond.GetBondType() == Chem.rdchem.BondType.SINGLE for bond in mol.GetBonds())
    if prop == "num_double_bonds":
        return sum(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE for bond in mol.GetBonds())
    if prop == "num_triple_bonds":
        return sum(bond.GetBondType() == Chem.rdchem.BondType.TRIPLE for bond in mol.GetBonds())
    if prop == "num_aromatic_bonds":
        return sum(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC for bond in mol.GetBonds())
    if prop == "num_carbon":
        return sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    if prop == "num_nitrogen":
        return sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms())
    if prop == "num_oxygen":
        return sum(atom.GetAtomicNum() == 8 for atom in mol.GetAtoms())
    if prop == "num_fluorine":
        return sum(atom.GetAtomicNum() == 9 for atom in mol.GetAtoms())
    if prop == "num_phosphorus":
        return sum(atom.GetAtomicNum() == 15 for atom in mol.GetAtoms())
    if prop == "num_sulfur":
        return sum(atom.GetAtomicNum() == 16 for atom in mol.GetAtoms())
    if prop == "num_chlorine":
        return sum(atom.GetAtomicNum() == 17 for atom in mol.GetAtoms())
    if prop == "num_bromine":
        return sum(atom.GetAtomicNum() == 35 for atom in mol.GetAtoms())
    if prop == "num_iodine":
        return sum(atom.GetAtomicNum() == 53 for atom in mol.GetAtoms())
    if prop == "num_boron":
        return sum(atom.GetAtomicNum() == 5 for atom in mol.GetAtoms())
    if prop == "num_silicon":
        return sum(atom.GetAtomicNum() == 14 for atom in mol.GetAtoms())
    if prop == "num_selenium":
        return sum(atom.GetAtomicNum() == 34 for atom in mol.GetAtoms())
    if prop == "num_tellurium":
        return sum(atom.GetAtomicNum() == 52 for atom in mol.GetAtoms())
    if prop == "num_arsenic":
        return sum(atom.GetAtomicNum() == 33 for atom in mol.GetAtoms())
    if prop == "num_antimony":
        return sum(atom.GetAtomicNum() == 51 for atom in mol.GetAtoms())
    if prop == "num_bismuth":
        return sum(atom.GetAtomicNum() == 83 for atom in mol.GetAtoms())
    if prop == "num_polonium":
        return sum(atom.GetAtomicNum() == 84 for atom in mol.GetAtoms())

    smarts_map = {
        "num_benzene_ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
        "num_hydroxyl": "[OX2H]",
        "num_anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
        "num_aldehyde": "[CX3H1](=O)[#6]",
        "num_ketone": "[#6][CX3](=O)[#6]",
        "num_carboxyl": "[CX3](=O)[OX2H1]",
        "num_ester": "[#6][CX3](=O)[OX2H0][#6]",
        "num_amide": "[NX3][CX3](=[OX1])[#6]",
        "num_amine": "[NX3;H2,H1;!$(NC=O)]",
        "num_nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "num_halo": "[F,Cl,Br,I]",
        "num_thioether": "[SX2][CX4]",
        "num_nitrile": "[NX1]#[CX2]",
        "num_thiol": "[#16X2H]",
        "num_sulfide": "[#16X2H0]",
        "num_disulfide": "[#16X2H0][#16X2H0]",
        "num_sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
        "num_sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
        "num_borane": "[BX3]",
    }
    if prop in smarts_map:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_map[prop]))
        if prop == "num_sulfide":
            disulfides = mol.GetSubstructMatches(Chem.MolFromSmarts("[#16X2H0][#16X2H0]"))
            return len(matches) - len(disulfides)
        return len(matches)

    raise ValueError(f"Unsupported property: {prop}")


def _calculate_similarity(smiles1: str, smiles2: str) -> float:
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


class SpeakToStructureMetric(EvaluateInstancesMetric):
    """RDKit-backed evaluator for the supported S2-Bench open-generation subtasks."""

    def _evaluate_success(self, task: str, subtask: str, row: Dict[str, Any], output_smiles: str, output_mol: Chem.Mol) -> Optional[int]:
        if output_mol is None:
            return 0

        if task == "MolCustom" and subtask == "AtomNum":
            for atom in _ATOM_FIELDS:
                if _mol_prop(output_mol, f"num_{atom}") != int(row.get(atom, 0)):
                    return 0
            return 1

        if task == "MolCustom" and subtask == "FunctionalGroup":
            for group in _FUNCTIONAL_GROUP_FIELDS:
                prop_name = "num_benzene_ring" if group == "benzene rings" else f"num_{group}"
                if _mol_prop(output_mol, prop_name) != int(row.get(group, 0)):
                    return 0
            return 1

        if task == "MolCustom" and subtask == "BondNum":
            mapping = {
                "single": "num_single_bonds",
                "double": "num_double_bonds",
                "triple": "num_triple_bonds",
                "rotatable": "rot_bonds",
                "aromatic": "num_aromatic_bonds",
            }
            for bond_name, prop_name in mapping.items():
                target = int(row.get(bond_name, 0))
                if target == 0:
                    continue
                if _mol_prop(output_mol, prop_name) != target:
                    return 0
            return 1

        if task == "MolEdit":
            raw_smiles = row["molecule"]
            raw_mol = Chem.MolFromSmiles(raw_smiles)
            if raw_mol is None:
                return None
            if subtask == "AddComponent":
                group = row["added_group"].replace("benzene ring", "benzene_ring")
                return int(_mol_prop(output_mol, f"num_{group}") == _mol_prop(raw_mol, f"num_{group}") + 1)
            if subtask == "DelComponent":
                group = row["removed_group"].replace("benzene ring", "benzene_ring")
                return int(_mol_prop(output_mol, f"num_{group}") == _mol_prop(raw_mol, f"num_{group}") - 1)
            if subtask == "SubComponent":
                added_group = row["added_group"].replace("benzene ring", "benzene_ring")
                removed_group = row["removed_group"].replace("benzene ring", "benzene_ring")
                removed_ok = _mol_prop(output_mol, f"num_{removed_group}") == _mol_prop(raw_mol, f"num_{removed_group}") - 1
                added_ok = _mol_prop(output_mol, f"num_{added_group}") == _mol_prop(raw_mol, f"num_{added_group}") + 1
                return int(removed_ok and added_ok)

        if task == "MolOpt":
            raw_smiles = row["molecule"]
            raw_mol = Chem.MolFromSmiles(raw_smiles)
            if raw_mol is None:
                return None
            instruction = str(row["Instruction"]).lower()
            lower_ok = "lower" in instruction or "decrease" in instruction
            prop_name = {"LogP": "logP", "MR": "MR", "QED": "qed"}[subtask]
            target_value = _mol_prop(output_mol, prop_name)
            raw_value = _mol_prop(raw_mol, prop_name)
            if target_value is None or raw_value is None:
                return 0
            if lower_ok:
                return int(target_value < raw_value)
            return int(target_value > raw_value)

        return None

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        valid_count = 0
        successes: List[int] = []
        similarities: List[float] = []

        for request_state in request_states:
            assert request_state.result is not None
            metadata = request_state.instance.extra_data or {}
            task = metadata.get("task")
            subtask = metadata.get("subtask")
            row = metadata.get("row", {})

            completion = request_state.result.completions[0].text
            output_smiles = _correct_text(completion)
            output_mol = _mol_from_output(completion)

            if output_mol is not None:
                valid_count += 1

            success = self._evaluate_success(task, subtask, row, output_smiles, output_mol)
            if success is not None:
                successes.append(success)

            if output_mol is not None and task in ("MolEdit", "MolOpt") and row.get("molecule"):
                similarities.append(_calculate_similarity(row["molecule"], output_smiles))

        total = len(request_states)
        validity = valid_count / total if total else 0.0
        success_rate = sum(successes) / len(successes) if successes else 0.0
        similarity = sum(similarities) / len(similarities) if similarities else 0.0

        return [
            Stat(MetricName("validity")).add(validity),
            Stat(MetricName("success_rate")).add(success_rate),
            Stat(MetricName("similarity")).add(similarity),
        ]
