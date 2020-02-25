const messages = (() => {
  const template = _.template(
    '<article class="message <%= direction %>">' +
      '<p><%= content %></p>' +
      '<footer>' +
        '<time datetime="<%= datetime %>"><%= now %></time>' +
      '</footer>' +
    '</article>'
  );

  const $messages = $('.messages');

  return {
    add: (direction, content) => {
      now = new Date();

      $messages.append(template({
        direction: direction,
        content: _.escape(content).replace(/\n/g, '<br>'),
        datetime: now.toISOString(),
        now: now.getHours().toString().padStart(2, '0') + ':' + now.getMinutes().toString().padStart(2, '0'),

      }));
      $messages.scrollTop($messages.prop('scrollHeight'));
    }
  }
})();

const ask = (utterance) => {
  return $.post('/agents/kepco', {utterance: utterance});
};

const $say = $('.say input');
const $morphemes = $('.logs .morphemes p');
const $domains = $('.logs .domains ul');
const $intents = $('.logs .intents ul');

$say.on('keyup', function (event) {
  if (event.which != 13) return;

  const content = $say.val().trim();
  if (content === '') return;

  messages.add('right', content);
  $say.val('');

  $say.prop('disabled', true);
  ask(content).then(data => {
    $morphemes.empty();
    for (let morpheme of data.morphemes) {
      $morphemes.append('<span>' + _.escape(morpheme) + '</span>');
    }

    $domains.empty();
    for (let domain of data.domains.slice(0, 5)) {
      $domains.append('<li>' + _.escape(domain.name) + ': ' + domain.confidence.toFixed(6) + '</li>');
    }

    $intents.empty();
    for (let intent of data.intents.slice(0, 5)) {
      $intents.append('<li>' + _.escape(intent.name) + ': ' + intent.confidence.toFixed(6) + '</li>');
    }

    if (data.responses.length === 0) {
      messages.add('left', '죄송합니다. 이해할 수 없는 질문입니다.');
      return;
    }

    for (let response of data.responses) {
      for (let action of response.actions) {
        messages.add('left', action.payload.text);
      }
    }
  }).then(() => {
    $say.prop('disabled', false);
    $say.focus();
  })

  return false;
});

$('.sample-sentences select ').change(function () {
  $say.val($(this).val());
  $say.focus();
});

$(document).on('click', '.message.right', function () {
  $say.val($(this).find('p').text());
  $say.focus();
});

messages.add('left', '안녕하세요. 궁금하신 사항을 질문해주세요.')
$say.focus();