import math
import edist.sed as sed
import edist.ted as ted
import numpy as np
import json
from tree_sitter import Language, Parser
from tqdm import trange


def api_var_edit_dis(hyps, refs, weight_dict):
    if len(refs) == 0 or len(hyps) == 0:
        return None
    score = np.zeros((len(hyps), len(refs)))
    for i in range(len(hyps)):
        for j in range(len(refs)):
            score[i][j] = 1 - sed.sed(hyps[i], refs[j]) / max(max(len(hyps[i]), len(refs[j])), 1)
    score = np.max(score, axis=1)
    weights = []
    for i in range(len(hyps)):
        try:
            weights.append(weight_dict[hyps[i]])
        except KeyError:
            weights.append(0)
    weights = np.array(weights) / np.linalg.norm(weights, ord=1)
    return np.average(score, weights=weights)


def tree_edit_dis(hyp, ref, parser):
    tree = parser.parse(bytes(hyp, 'utf8')).root_node
    all_nodes1 = []
    get_all_subtrees(tree, all_nodes1)
    hyp_nodes, hyp_adj = extract_subtrees(all_nodes1[0])

    tree = parser.parse(bytes(ref, 'utf8')).root_node
    all_nodes2 = []
    get_all_subtrees(tree, all_nodes2)
    ref_nodes, ref_adj = extract_subtrees(all_nodes2[0])

    tree_dis = ted.standard_ted(hyp_nodes, hyp_adj, ref_nodes, ref_adj)

    return tree_dis / max(len(hyp_nodes), len(ref_nodes))


def extract_subtrees(tree):
    nodes, adj_nodes = [], []
    name_dict, num_dict = {}, {}
    traverse_tree(tree, nodes, adj_nodes, name_dict, num_dict)

    nodes = [name_dict[node] for node in nodes]
    children_index = []
    for children in adj_nodes:
        tmp = []
        for node in children:
            tmp.append(num_dict[node])
        children_index.append(tmp)
    return nodes, children_index


def get_all_subtrees(tree, nodes):
    if tree.children is not None:
        nodes.append(tree)

    for child in tree.children:
        get_all_subtrees(child, nodes)


def traverse_tree(tree, nodes, adj_nodes, name_dict=None, num_dict=None):
    if tree.children is not None:
        nodes.append(tree.id)
        adj_nodes.append([node.id for node in tree.children])
        name_dict[tree.id] = tree.type
        num_dict[tree.id] = len(num_dict)

    for child in tree.children:
        traverse_tree(child, nodes, adj_nodes, name_dict, num_dict)


def extract_var(codes, parser, lang):
    vnames = []
    for i in range(len(codes)):
        tmp = set()
        tree = parser.parse(bytes(codes[i], 'utf8'))
        if lang == 'python':
            _get_var_names_from_node_python(tree.root_node, tmp, codes[i])
        elif lang == 'java':
            _get_var_names_from_node_java(tree.root_node, tmp, codes[i])
        vnames.append(list(tmp))

    return vnames


def _get_var_names_from_node_python(node, vnames, code, inactive=False):
    if len(node.children) > 0:
        for child in node.children:
            if (
                    (node.type == "call" and child.type != "argument_list") or
                    (node.type == "attribute") or
                    (node.type in ["import_statement", "import_from_statement"])
            ):
                _get_var_names_from_node_python(child, vnames, code, True)
            else:
                _get_var_names_from_node_python(child, vnames, code, inactive)
    elif node.type == "identifier":
        if not inactive:
            vnames.add(span_select(node, code=code))


def _get_var_names_from_node_java(node, vnames, code, inactive=False):
    if len(node.children) > 0:
        if node.type in ["method_invocation"]:
            _get_var_names_from_node_java(node.children[0], vnames, code, inactive)
            for child in node.children[1:]:
                if child.type == "argument_list":
                    _get_var_names_from_node_java(child, vnames, code, inactive)
                else:
                    _get_var_names_from_node_java(child, vnames, code, True)
        else:
            for child in node.children:
                if node.type in ["field_access", "modifiers", "import_declaration", "package_declaration"]:
                    _get_var_names_from_node_java(child, vnames, code, True)
                else:
                    _get_var_names_from_node_java(child, vnames, code, inactive)
    elif node.type == "identifier":
        if not inactive:
            vnames.add(span_select(node, code=code))


def extract_api(codes, parser, lang):
    api_seq = []
    for i in range(len(codes)):
        tmp = []
        tree = parser.parse(bytes(codes[i], 'utf8'))
        if lang == 'python':
            _get_api_seq_python(codes[i], tree.root_node, tmp)
        elif lang == 'java':
            _get_api_seq_java(codes[i], tree.root_node, tmp)
        tmp = set(tmp)
        api_seq.append(list(tmp))

    return api_seq


def _get_api_seq_python(code, node, api_seq, tmp=None):
    if node.type == "call":
        api = node.child_by_field_name("function")
        if tmp:
            tmp.append(span_select(api, code=code))
            ant = False
        else:
            tmp = [span_select(api, code=code)]
            ant = True
        for child in node.children:
            _get_api_seq_python(code, child, api_seq, tmp)
        if ant:
            api_seq += tmp[::-1]
            tmp = None
    else:
        for child in node.children:
            _get_api_seq_python(code, child, api_seq, tmp)


def _get_api_seq_java(code, node, api_seq):
    if node.type == "method_invocation":
        obj = node.child_by_field_name("object")
        func = node.child_by_field_name("name")
        if obj:
            api_seq.append(span_select(obj, code=code) + "." + span_select(func, code=code))
        else:
            api_seq.append(span_select(func, code=code))
    else:
        for child in node.children:
            _get_api_seq_java(code, child, api_seq)


def span_select(*nodes, code, indent=False):
    if not nodes:
        return ""
    start, end = nodes[0].start_byte, nodes[-1].end_byte
    select = code[start:end]
    if indent:
        return " " * nodes[0].start_point[1] + select
    return select


def api_postprocess(api_list):
    final_api = []
    for api in api_list:
        if '.' in api:
            final_api.append(api.split('.')[-1])
        else:
            final_api.append(api)
    return list(set(final_api))


def var_postprocess(var_list):
    final_var = []
    for var in var_list:
        if '_' in var:
            final_var.extend(var.split('_'))
        else:
            final_var.append(var)
    return list(set(final_var))


def cal_codestyle_dis(hyp, ref, lang, weight_var, weight_api):
    JAVA_LANGUAGE = Language('./parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)

    hyp_api = extract_api([hyp], parser, lang)[0]
    hyp_api = api_postprocess(hyp_api)
    ref_api = extract_api([ref], parser, lang)[0]
    ref_api = api_postprocess(ref_api)
    if len(hyp_api) == 0 or len(ref_api) == 0:
        api_dis = None
    else:
        api_dis = min(api_var_edit_dis(hyp_api, ref_api, weight_api), api_var_edit_dis(ref_api, hyp_api, weight_api))

    hyp_var = extract_var([hyp], parser, lang)[0]
    hyp_var = var_postprocess(hyp_var)
    ref_var = extract_var([ref], parser, lang)[0]
    ref_var = var_postprocess(ref_var)
    if len(hyp_var) == 0 or len(ref_var) == 0:
        var_dis = None
    else:
        var_dis = min(api_var_edit_dis(hyp_var, ref_var, weight_var), api_var_edit_dis(ref_var, hyp_var, weight_var))

    tree_dis = 1 - tree_edit_dis(hyp, ref, parser)

    return api_dis, var_dis, tree_dis


def get_overall_csd(code, aug, lang, weight_var, weight_api):
    api_dis, var_dis, tree_dis = cal_codestyle_dis(code, aug, lang, weight_var, weight_api)
    if api_dis is None and var_dis is not None:
        dis = (var_dis + tree_dis) / 2
    elif api_dis is not None and var_dis is None:
        dis = (api_dis + tree_dis) / 2
    elif api_dis is None and var_dis is None:
        dis = tree_dis
    else:
        dis = (api_dis + var_dis + tree_dis) / 3

    return dis


def cal_idf(hyps, refs, phase, lang):
    count = {}
    for code in hyps:
        if phase == 'var':
            vars = extract_var([code], parser, lang)[0]
            items = var_postprocess(vars)
        elif phase == 'api':
            apis = extract_api([code], parser, lang)[0]
            items = api_postprocess(apis)
        for item in items:
            if item in count:
                count[item] += 1
            else:
                count[item] = 1
    for code in refs:
        if phase == 'var':
            vars = extract_var([code], parser, lang)[0]
            items = var_postprocess(vars)
        elif phase == 'api':
            apis = extract_api([code], parser, lang)[0]
            items = api_postprocess(apis)
        for item in items:
            if item in count:
                count[item] += 1
            else:
                count[item] = 1

    total = 0
    for value in count.values():
        total += value
    for key in count.keys():
        count[key] = math.log(total / count[key], math.e)

    return count


if __name__ == '__main__':
    pl = {'conala': 'python',
          'mbpp': 'python',
          'apps': 'python',
          'mbjp': 'java'}

    for dataset in ['mbpp', 'apps', 'mbjp', 'conala']:
        for modelsize in ['7b', '13b', '34b', 'gpt35']:

            with open('../data/{}/aug-code-{}.json'.format(dataset, modelsize), 'r') as f:
                code_gen_aug = json.load(f)
                code_gen_aug = code_gen_aug['test']
            with open('../data/{}/aug-query-{}.json'.format(dataset, modelsize), 'r') as f:
                query_gen_aug = json.load(f)
                query_gen_aug = query_gen_aug['test']

            JAVA_LANGUAGE = Language('./parser/my-languages.so', pl[dataset])
            parser = Parser()
            parser.set_language(JAVA_LANGUAGE)

            score = np.zeros(len(query_gen_aug) // 4)

            var_idf = cal_idf(code_gen_aug, query_gen_aug, 'var', pl[dataset])
            api_idf = cal_idf(code_gen_aug, query_gen_aug, 'api', pl[dataset])

            for i in trange(len(query_gen_aug) // 4):
                tmp_score = []
                for j in range(4):
                    tmp_score.append(get_overall_csd(code_gen_aug[4 * i + j], query_gen_aug[4 * i + j], pl[dataset], var_idf, api_idf))
                score[i] = np.mean(tmp_score).item()

            print('{} {} CSD score: '.format(dataset, modelsize), np.round(np.average(score), 3))

