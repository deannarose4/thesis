document.addEventListener('DOMContentLoaded', function() {
  // Initialize sidenav
  var sidenav = document.querySelectorAll('.sidenav');
  var sidenavInstance = M.Sidenav.init(sidenav, {
    edge: 'right',
    draggable: false,
    inDuration: 250,
    outDuration: 200,
    onOpenStart: function(el) {
      el.classList.add('show');
    },
    onCloseStart: function(el) {
      el.classList.remove('show');
    }
  });

  // Handle sidenav trigger click
  document.querySelector('.sidenav-trigger').addEventListener('click', function(e) {
    e.preventDefault();
    var instance = M.Sidenav.getInstance(document.querySelector('.sidenav'));
    if (instance) {
      instance.open();
    }
  });

  // Close sidenav when clicking outside
  document.addEventListener('click', function(e) {
    if (!e.target.closest('.sidenav') && !e.target.closest('.sidenav-trigger')) {
      var instance = M.Sidenav.getInstance(document.querySelector('.sidenav'));
      if (instance && instance.isOpen) {
        instance.close();
      }
    }
  });

  // Close sidenav when clicking a link
  document.querySelectorAll('.sidenav a').forEach(function(link) {
    link.addEventListener('click', function() {
      var instance = M.Sidenav.getInstance(document.querySelector('.sidenav'));
      if (instance) {
        instance.close();
      }
    });
  });
});
