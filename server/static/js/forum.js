function post_data(url, data) {
    return fetch(url, {
      body: JSON.stringify(data),
      cache: 'no-cache',
      credentials: 'same-origin',
      headers: {
        'user-agent': 'Mozilla/4.0 MDN Example',
        'content-type': 'application/json'
      },
      method: 'POST',
      mode: 'cors',
      redirect: 'follow',
      referrer: 'no-referrer',
    })
    .then(response => response.json())
}

function post_comment() {
    var data = {
        video_id: parseInt(document.getElementById("video_id").value), 
        comment: document.getElementById("comment").value, 
        tag: document.getElementById("tag").value
    };

    console.log(data);
    post_data("/post", data);
}

