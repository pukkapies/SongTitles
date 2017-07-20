from training_utils import add_to_collection


def create_CharRNN_collections(CharRNN_obj, restore_key):
    CharRNN_obj.collections[restore_key + 'input_ph'] = CharRNN_obj.input_ph
    CharRNN_obj.collections[restore_key + 'targets'] = CharRNN_obj.targets
    CharRNN_obj.collections[restore_key + 'targets_one_hot'] = CharRNN_obj.targets_one_hot
    CharRNN_obj.collections[restore_key + 'loss_mask'] = CharRNN_obj.loss_mask
    CharRNN_obj.collections[restore_key + 'batch_size'] = CharRNN_obj.batch_size
    CharRNN_obj.collections[restore_key + 'sequence_length'] = CharRNN_obj.sequence_length
    CharRNN_obj.collections[restore_key + 'initial_state'] = CharRNN_obj.initial_state
    CharRNN_obj.collections[restore_key + 'embedding'] = CharRNN_obj.embedding
    CharRNN_obj.collections[restore_key + 'flat_logits'] = CharRNN_obj.flat_logits
    CharRNN_obj.collections[restore_key + 'flat_probs'] = CharRNN_obj.flat_probs
    CharRNN_obj.collections[restore_key + 'probs'] = CharRNN_obj.probs
    CharRNN_obj.collections[restore_key + 'masked_loss'] = CharRNN_obj.masked_loss
    CharRNN_obj.collections[restore_key + 'loss'] = CharRNN_obj.loss

    add_to_collection(**CharRNN_obj.collections)


def unpack_CharRNN_handles(ResNetVAE_obj, restore_key, session):
    ResNetVAE_obj.input_ph = session.graph.get_collection(restore_key + 'input_ph')[0]
    ResNetVAE_obj.targets = session.graph.get_collection(restore_key + 'targets')[0]
    ResNetVAE_obj.targets_one_hot = session.graph.get_collection(restore_key + 'targets_one_hot')[0]
    ResNetVAE_obj.loss_mask = session.graph.get_collection(restore_key + 'loss_mask')[0]
    ResNetVAE_obj.batch_size = session.graph.get_collection(restore_key + 'batch_size')[0]
    ResNetVAE_obj.sequence_length = session.graph.get_collection(restore_key + 'sequence_length')[0]
    ResNetVAE_obj.initial_state = session.graph.get_collection(restore_key + 'initial_state')[0]
    ResNetVAE_obj.embedding = session.graph.get_collection(restore_key + 'embedding')[0]
    ResNetVAE_obj.flat_logits = session.graph.get_collection(restore_key + 'flat_logits')[0]
    ResNetVAE_obj.flat_probs = session.graph.get_collection(restore_key + 'flat_probs')[0]
    ResNetVAE_obj.probs = session.graph.get_collection(restore_key + 'probs')[0]
    ResNetVAE_obj.masked_loss = session.graph.get_collection(restore_key + 'masked_loss')[0]
    ResNetVAE_obj.loss = session.graph.get_collection(restore_key + 'loss')[0]
